//! Side-by-side evaluation of burst detectors through the FM channel simulator.
//!
//! Every detector sees exactly the same received audio, so differences are paired
//! and can be tested with McNemar's exact test instead of comparing two noisy rates.
//! For each CNR it reports, per detector:
//! - `hit`: the detector found the burst within ±ncp/4 samples of the true preamble B;
//! - `burst`: the whole burst (header and every codeword) decoded from where the
//!   detector pointed, as the streaming receiver would decode it.
//!
//! `genie` decodes from the true position. It is the ceiling for any detector: the
//! gap between `genie` and a detector is what detection costs at that CNR.
//!
//! A separate pass counts detections and decoded headers per minute on audio that
//! holds no burst (squelch noise, a weak unmodulated carrier, tones, white noise).
//! A detection there costs a header decode; a decoded header there is a false burst.
//!
//! `cargo test --release -p vardefm-modem detector_eval -- --ignored --nocapture`
//!
//! Environment: `MODES` (narrow,wide), `TRIALS` per CNR (100), `LEVEL` (1), `NCW` (1),
//! `CNRS` (dB list, default per mode), `NOISE_SECONDS` per noise case (60),
//! `CSV` (file to append the burst results to).

use super::*;
use std::f32::consts::PI;
use std::io::Write;
use std::sync::Arc;
use vardefm_dsp::prng::Prng;
use vardefm_sim::{transmit, FmChannelConfig};

/// A burst detector: search `rx` from `from`, return where the body of preamble B starts.
pub(crate) type DetectFn = fn(&Modem, &[f32], usize) -> Option<usize>;

/// Detectors under evaluation. The first one is the reference for the paired tests.
pub(crate) const DETECTORS: &[(&str, DetectFn)] = &[("sc", |m, rx, from| m.ofdm.detect(rx, from))];

/// Leader length used by the evaluation, seconds (the `vardefm modem` default).
const TXDELAY: f32 = 0.15;

fn modem(mode: &str) -> Arc<Modem> {
    let mut m = Modem::new(OfdmParams::by_name(mode).unwrap(), TXDELAY);
    // Trials already run on every core.
    m.decode_threads = 1;
    Arc::new(m)
}

fn burst(level: u8, ncw: usize, seed: u32) -> Burst {
    let mut rng = Prng::new(seed);
    let n = Modem::payload_bytes(level);
    Burst {
        header: Header { kind: Kind::Data, session: 0x42, level, ncw: ncw as u8, ..Default::default() },
        payload: (0..ncw).map(|_| (0..n).map(|_| rng.next_u32() as u8).collect()).collect(),
    }
}

/// Decode the burst whose preamble B starts at `b` the way `Receiver` does, and
/// say whether header and every codeword came out as sent.
fn decodes_at(m: &Modem, rx: &[f32], b: usize, sent: &Burst) -> bool {
    let hdr_syms = m.header_symbols();
    let Some(h) = m.ofdm.demodulate(rx, b, hdr_syms).and_then(|f| m.decode_header(&f)) else {
        return false;
    };
    if h != sent.header {
        return false;
    }
    if h.ncw == 0 {
        return true;
    }
    let nsym = hdr_syms + h.ncw as usize * m.symbols_per_codeword(h.level);
    let Some(frame) = m.ofdm.demodulate_known(rx, b, nsym, &m.header_carriers(&h)) else {
        return false;
    };
    m.decode_payload(&h, &frame).into_iter().zip(&sent.payload).all(|(got, want)| got.as_ref() == Some(want))
}

/// Outcome of one trial: (hit, burst) per detector, then the genie's burst.
struct Trial {
    det: Vec<(bool, bool)>,
    genie: bool,
}

fn trial(m: &Modem, mode: &str, cnr: f32, sent: &Burst, tx: &[f32], seed: u64) -> Trial {
    let cfg = FmChannelConfig { cnr_db: Some(cnr), seed, ..FmChannelConfig::by_name(mode).unwrap() };
    // The channel's delays do not depend on the noise, so the clean copy gives the truth.
    let truth = m.ofdm.detect(&transmit(&FmChannelConfig { cnr_db: None, ..cfg.clone() }, tx), 0);
    let truth = truth.expect("burst not found on a clean channel");
    let rx = transmit(&cfg, tx);
    let tol = m.ofdm.p.ncp / 4;
    let det = DETECTORS
        .iter()
        .map(|(_, f)| match f(m, &rx, 0) {
            Some(b) => (b.abs_diff(truth) <= tol, decodes_at(m, &rx, b, sent)),
            None => (false, false),
        })
        .collect();
    Trial { det, genie: decodes_at(m, &rx, truth, sent) }
}

/// Run `trials` trials on every core. Trial `t` uses channel seed `1000 + t`.
fn run_trials(m: &Modem, mode: &str, cnr: f32, sent: &Burst, trials: u32) -> Vec<Trial> {
    let tx = m.modulate(sent);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u32;
    let mut out: Vec<(u32, Trial)> = std::thread::scope(|s| {
        let tx = &tx;
        let workers: Vec<_> = (0..threads)
            .map(|th| {
                s.spawn(move || {
                    (th..trials)
                        .step_by(threads as usize)
                        .map(|t| (t, trial(m, mode, cnr, sent, tx, 1000 + t as u64)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    out.sort_by_key(|(t, _)| *t);
    out.into_iter().map(|(_, r)| r).collect()
}

/// Two-sided exact McNemar p-value for `b` and `c` discordant pairs.
fn mcnemar(b: u32, c: u32) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut term = 0.5f64.powi(n as i32);
    let mut tail = 0.0;
    for i in 0..=k {
        tail += term;
        term *= (n - i) as f64 / (i + 1) as f64;
    }
    (2.0 * tail).min(1.0)
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Audio with no burst in it, `seconds` long, for the false-detection pass.
fn noise_cases(mode: &str, seconds: f32) -> Vec<(&'static str, Vec<f32>)> {
    let n = (seconds * 48_000.0) as usize;
    let base = FmChannelConfig { keyup_delay_s: 0.0, ..FmChannelConfig::by_name(mode).unwrap() };
    let mut rng = Prng::new(99);
    let white: Vec<f32> = (0..n).map(|_| (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.6).collect();
    let tones: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.15 * ((2.0 * PI * 750.0 * t).sin() + (2.0 * PI * 1500.0 * t).sin())
        })
        .collect();
    vec![
        ("squelch open, no carrier", transmit(&FmChannelConfig { cnr_db: Some(-10.0), ..base.clone() }, &vec![0.0; n])),
        ("weak carrier, 5 dB CNR", transmit(&FmChannelConfig { cnr_db: Some(5.0), ..base }, &vec![0.0; n])),
        ("white noise", white),
        ("750 + 1500 Hz tones", tones),
    ]
}

/// Detections and decoded headers in `rx`, scanning on after each detection.
fn false_detections(m: &Modem, f: DetectFn, rx: &[f32]) -> (u32, u32) {
    let (mut dets, mut headers) = (0, 0);
    let mut pos = 0;
    while let Some(b) = f(m, rx, pos) {
        dets += 1;
        if m.ofdm.demodulate(rx, b, m.header_symbols()).and_then(|fr| m.decode_header(&fr)).is_some() {
            headers += 1;
        }
        pos = b + 1;
    }
    (dets, headers)
}

#[test]
#[ignore]
fn detector_eval() {
    let modes: String = env_or("MODES", "narrow,wide".to_string());
    let trials: u32 = env_or("TRIALS", 100);
    let level: u8 = env_or("LEVEL", 1);
    let ncw: usize = env_or("NCW", 1);
    let noise_s: f32 = env_or("NOISE_SECONDS", 60.0);
    let mut csv = std::env::var("CSV").ok().map(|p| {
        let new = !std::path::Path::new(&p).exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
        if new {
            writeln!(f, "mode,level,ncw,cnr_db,trials,detector,hits,bursts").unwrap();
        }
        f
    });
    let names: Vec<&str> = DETECTORS.iter().map(|(n, _)| *n).collect();
    println!(
        "level {level} ({}), {ncw} codeword(s), {trials} trials per point, txdelay {TXDELAY} s",
        LEVELS[level as usize].name
    );
    for mode in modes.split(',') {
        let m = modem(mode);
        let sent = burst(level, ncw, 7);
        let default_cnrs = if mode == "narrow" { "3,4,5,6,7,8,10" } else { "5,6,7,8,9,10,12" };
        let cnrs: String = env_or("CNRS", default_cnrs.to_string());
        println!("\n{mode}: hits and decoded bursts out of {trials}; paired test against '{}'", names[0]);
        for cnr in cnrs.split(',').map(|c| c.trim().parse::<f32>().unwrap()) {
            let res = run_trials(&m, mode, cnr, &sent, trials);
            let genie = res.iter().filter(|r| r.genie).count();
            let mut line = format!("  CNR {cnr:5.1} dB  genie {genie:4}");
            for (d, name) in names.iter().enumerate() {
                let hits = res.iter().filter(|r| r.det[d].0).count();
                let bursts = res.iter().filter(|r| r.det[d].1).count();
                line += &format!(" | {name} hit {hits:4} burst {bursts:4}");
                if d > 0 {
                    let better = res.iter().filter(|r| r.det[d].1 && !r.det[0].1).count() as u32;
                    let worse = res.iter().filter(|r| !r.det[d].1 && r.det[0].1).count() as u32;
                    line += &format!(" (+{better}/-{worse}, p {:.3})", mcnemar(better, worse));
                }
                if let Some(f) = &mut csv {
                    writeln!(f, "{mode},{level},{ncw},{cnr},{trials},{name},{hits},{bursts}").unwrap();
                }
            }
            if let Some(f) = &mut csv {
                writeln!(f, "{mode},{level},{ncw},{cnr},{trials},genie,{trials},{genie}").unwrap();
            }
            println!("{line}");
        }
        println!("{mode}: per minute of audio without a burst, detections / decoded headers");
        for (case, audio) in noise_cases(mode, noise_s) {
            let mut line = format!("  {case:26}");
            for (name, f) in DETECTORS {
                let (d, h) = false_detections(&m, *f, &audio);
                let per_min = 60.0 / noise_s;
                line += &format!(" | {name} {:7.2} / {:5.2}", d as f32 * per_min, h as f32 * per_min);
            }
            println!("{line}");
        }
    }
}

/// The harness itself: on a strong signal every detector and the genie decode every
/// burst, and the reference detector raises no false headers on squelch noise.
#[test]
fn harness_sees_strong_bursts_and_no_false_headers() {
    for mode in ["narrow", "wide"] {
        let m = modem(mode);
        let sent = burst(1, 1, 7);
        for r in run_trials(&m, mode, 25.0, &sent, 4) {
            assert!(r.genie, "{mode}: genie failed on a strong signal");
            assert!(r.det.iter().all(|&(hit, ok)| hit && ok), "{mode}: a detector missed a strong burst");
        }
        let (_, audio) = noise_cases(mode, 2.0).swap_remove(0);
        for (name, f) in DETECTORS {
            assert_eq!(false_detections(&m, *f, &audio).1, 0, "{mode}: {name} decoded a header from noise");
        }
    }
}

#[test]
fn mcnemar_matches_known_values() {
    assert_eq!(mcnemar(0, 0), 1.0);
    assert!((mcnemar(0, 5) - 0.0625).abs() < 1e-12);
    assert!((mcnemar(2, 8) - 0.109375).abs() < 1e-12);
    assert_eq!(mcnemar(4, 4), 1.0);
}

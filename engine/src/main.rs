use illusion_engine::uci;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(9);
        let (nodes, millis) = uci::bench(depth);
        let nps = nodes as u128 * 1000 / millis.max(1);
        println!("{nodes} nodes {millis} ms {nps} nps");
    } else {
        uci::run();
    }
}

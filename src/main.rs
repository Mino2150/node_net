use node_net::run;

fn main() {
    pollster::block_on(run());
}

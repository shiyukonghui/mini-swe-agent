#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = mini_swe_agent::run::Cli::parse();
    mini_swe_agent::run::run(cli).await
}

# Battery Thermal RL - Makefile
# Convenient commands for common operations

.PHONY: help install test clean demo train evaluate plot

# Default target
help:
	@echo "Battery Thermal RL - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies"
	@echo "  test        Test installation"
	@echo ""
	@echo "Data & Simulation:"
	@echo "  climate     Generate climate data for São Paulo (30 days)"
	@echo "  battery     Simulate battery at different temperatures"
	@echo "  industrial  Simulate industrial system (7 days)"
	@echo ""
	@echo "Reinforcement Learning:"
	@echo "  train       Train PPO agent (quick training)"
	@echo "  train-full  Train PPO agent (full training)"
	@echo "  evaluate    Evaluate trained agent"
	@echo ""
	@echo "Analysis:"
	@echo "  demo        Run complete demonstration"
	@echo "  report      Generate analysis report"
	@echo "  plot        Generate plots (if data available)"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       Clean output files"
	@echo "  info        Show system information"

# Setup commands
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

test:
	@echo "Testing installation..."
	python cli.py info status
	@echo "Test complete!"

# Data generation commands
climate:
	@echo "Generating climate data for São Paulo (30 days)..."
	python cli.py climate generate --region southeast_sp --days 30 --output data/sp_climate_30d.csv
	@echo "Climate data generated!"

climate-all:
	@echo "Generating climate data for all regions (7 days each)..."
	python cli.py climate generate --region southeast_sp --days 7 --output data/southeast_sp_7d.csv
	python cli.py climate generate --region northeast_rn --days 7 --output data/northeast_rn_7d.csv
	python cli.py climate generate --region south_rs --days 7 --output data/south_rs_7d.csv
	python cli.py climate generate --region central_west_mt --days 7 --output data/central_west_mt_7d.csv
	python cli.py climate generate --region north_am --days 7 --output data/north_am_7d.csv
	@echo "All climate data generated!"

# Battery simulation
battery:
	@echo "Simulating Li-ion battery at different temperatures..."
	python cli.py battery simulate --type li_ion_500kwh --temperature 15 --action charge --hours 4
	python cli.py battery simulate --type li_ion_500kwh --temperature 25 --action charge --hours 4
	python cli.py battery simulate --type li_ion_500kwh --temperature 35 --action charge --hours 4
	python cli.py battery simulate --type li_ion_500kwh --temperature 45 --action charge --hours 4

battery-compare:
	@echo "Comparing battery types..."
	python cli.py battery compare --temperature-range 10 50 --output outputs/battery_comparison.csv
	@echo "Battery comparison saved to outputs/battery_comparison.csv"

# Industrial system simulation
industrial:
	@echo "Simulating industrial system (7 days)..."
	python cli.py industrial simulate --profile medium_metallurgy --days 7 --output outputs/industrial_sim_7d.csv
	@echo "Industrial simulation saved to outputs/industrial_sim_7d.csv"

industrial-economics:
	@echo "Running economic analysis (30 days)..."
	python cli.py industrial economics --profile medium_metallurgy --days 30

# RL training commands
train:
	@echo "Training PPO agent (quick training - 10k steps)..."
	python cli.py rl train --algorithm ppo --steps 10000 --output models/ppo_quick --eval-freq 2000
	@echo "Quick training complete! Model saved to models/ppo_quick.zip"

train-full:
	@echo "Training PPO agent (full training - 100k steps)..."
	python cli.py rl train --algorithm ppo --steps 100000 --output models/ppo_full --eval-freq 10000 --tensorboard-log logs/
	@echo "Full training complete! Model saved to models/ppo_full.zip"

train-sac:
	@echo "Training SAC agent (50k steps)..."
	python cli.py rl train --algorithm sac --steps 50000 --output models/sac_agent --eval-freq 5000

evaluate:
	@echo "Evaluating trained agent..."
	@if [ -f models/ppo_quick.zip ]; then \
		python cli.py rl evaluate --model models/ppo_quick.zip --episodes 5 --output outputs/evaluation_results.json; \
		echo "Evaluation results saved to outputs/evaluation_results.json"; \
	elif [ -f models/ppo_full.zip ]; then \
		python cli.py rl evaluate --model models/ppo_full.zip --episodes 5 --output outputs/evaluation_results.json; \
		echo "Evaluation results saved to outputs/evaluation_results.json"; \
	else \
		echo "No trained model found! Run 'make train' first."; \
	fi

# Testing
demo:
	@echo "Running system demonstration..."
	python cli.py info status
	python cli.py climate generate --region southeast_sp --days 7 --output data/demo_climate.csv
	python cli.py battery simulate --type li_ion_500kwh --temperature 25 --hours 4

test-env:
	@echo "Testing RL environment..."
	python cli.py rl test --steps 100 --random-actions

# Analysis commands
report:
	@echo "Generating analysis report..."
	python cli.py analysis report --config config.json --output outputs/analysis_report.html
	@echo "Report saved to outputs/analysis_report.html"

optimize:
	@echo "Running optimization analysis..."
	python cli.py analysis optimize --battery-type li_ion_500kwh --industrial-profile medium_metallurgy --days 30

plot:
	@echo "Generating plots..."
	@if [ -f data/sp_climate_30d.csv ]; then \
		python cli.py analysis plot --type climate --data data/sp_climate_30d.csv --output outputs/climate_plot.png; \
		echo "Climate plot saved to outputs/climate_plot.png"; \
	else \
		echo "No climate data found! Run 'make climate' first."; \
	fi
	@if [ -f outputs/industrial_sim_7d.csv ]; then \
		python cli.py analysis plot --type industrial --data outputs/industrial_sim_7d.csv --output outputs/industrial_plot.png; \
		echo "Industrial plot saved to outputs/industrial_plot.png"; \
	fi

# Information commands
info:
	python cli.py info status

list:
	python cli.py info list

config:
	python cli.py info config

regions:
	python cli.py climate list

# Maintenance
clean:
	@echo "Cleaning output files..."
	rm -rf outputs/*
	rm -rf data/*
	rm -rf models/*
	rm -rf logs/*
	@echo "Clean complete!"

clean-models:
	@echo "Cleaning trained models..."
	rm -rf models/*
	rm -rf logs/*

# Development shortcuts
dev-setup: install test climate battery-compare industrial

dev-quick: dev-setup train evaluate report
	@echo "Quick development cycle complete!"

dev-full: dev-setup train-full evaluate report plot
	@echo "Full development cycle complete!"

# Benchmark different configurations
benchmark:
	@echo "Running benchmark across different configurations..."
	@echo "This will take several hours..."
	python cli.py rl train --algorithm ppo --battery-type li_ion_100kwh --steps 50000 --output models/bench_100kwh_ppo
	python cli.py rl train --algorithm ppo --battery-type li_ion_500kwh --steps 50000 --output models/bench_500kwh_ppo
	python cli.py rl train --algorithm sac --battery-type li_ion_500kwh --steps 50000 --output models/bench_500kwh_sac
	@echo "Benchmark training complete!"

# Help with specific commands
help-cli:
	python cli.py --help

help-climate:
	python cli.py climate --help

help-battery:
	python cli.py battery --help

help-rl:
	python cli.py rl --help

help-analysis:
	python cli.py analysis --help
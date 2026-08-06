# AI Dynamic Traffic Signal Control Analytics

This project is a data analytics and A/B testing case study for an intelligent traffic signal system. It compares a static fixed-time traffic light controller against a reinforcement-learning-based dynamic controller and presents the results through statistical analysis, computer vision traffic logs, environmental impact estimates, and an interactive dashboard.

The goal is to show how AI-driven signal timing can reduce congestion, improve vehicle throughput, and lower emissions at an urban intersection.

## Project Highlights

- Compared a `Static_120s_Fixed` signal plan with an `RL_PPO_Dynamic` traffic signal controller.
- Analyzed 7,200 seconds of signal-performance data across both controller strategies.
- Processed 4,857 vehicle detection records from YOLO-style computer vision logs.
- Built an executive dashboard for visualizing wait time, queue length, throughput, CO2 emissions, vehicle mix, lane density, and signal phase behavior.
- Used Welch's t-test and Cohen's d to validate whether the RL controller significantly improved average wait time.
- Estimated annual environmental and fuel-saving impact for city-scale deployment.

## Key Results

| Metric | Static Fixed Timer | RL Dynamic Controller | Improvement |
|---|---:|---:|---:|
| Average wait time | 34.14 sec | 6.13 sec | 82.0% lower |
| Average queue length | 11.79 cars | 3.59 cars | 69.6% lower |
| Average throughput | 1.99 vehicles/sec | 3.52 vehicles/sec | 76.9% higher |
| Total CO2, 1 hour | 33.946 kg | 10.394 kg | 23.55 kg saved/hr |
| Annualized CO2 saving | - | - | 206.3 metric tons/intersection |
| Estimated fuel saving | - | - | 89,314 liters/year |

Statistical test result:

- Test used: Welch's independent two-sample t-test
- Null hypothesis: Static and RL controllers have equal average wait times
- p-value: less than 0.0001
- Cohen's d: 2.23
- Interpretation: The RL controller produced a statistically significant and very large reduction in average waiting time.

## Dashboard Preview

Open `traffic_signal_dashboard.html` to view the dashboard. It includes:

- KPI summary cards for wait-time reduction, queue reduction, throughput gain, and CO2 savings
- A/B wait-time trajectory comparison
- Queue-length trend comparison
- Throughput analysis
- Welch's t-test and confidence interval visualization
- Vehicle class distribution from detection logs
- Estimated speed by vehicle type
- Lane-by-lane density distribution
- CO2 emissions comparison
- City-scale ROI estimator
- Interactive intersection phase simulator

## Repository Structure

```text
traffic-rl-project/
|-- README.md
|-- requirements.txt
|-- eda_traffic_analysis.py
|-- traffic_signal_dashboard.html
|-- dashboard_summary.json
|-- traffic_detections_log.csv
`-- signal_performance_comparison.csv
```

## File Descriptions

### `eda_traffic_analysis.py`

Main analytics script. It loads the detection and performance datasets, compares the two controller strategies, calculates statistical significance, computes business and environmental impact metrics, and exports `dashboard_summary.json`.

Main tasks performed:

- Loads raw vehicle detections and signal performance logs
- Splits data into static-control and RL-control groups
- Calculates mean wait time, queue length, throughput, emissions, and confidence intervals
- Runs Welch's t-test on average wait time
- Calculates Cohen's d effect size
- Computes CO2, fuel, and annualized impact estimates
- Aggregates time-series data into 30-second dashboard buckets
- Summarizes computer vision metrics such as vehicle mix, lane volume, camera volume, emergency vehicle count, confidence score, and speed by vehicle class

### `traffic_detections_log.csv`

Computer vision detection log. Each row represents a vehicle detected from an intersection camera.

Important columns:

- `timestamp`: detection timestamp
- `camera_id`: camera source
- `vehicle_class`: detected class such as Car, Motorcycle, Bus, Truck, Rickshaw, or Ambulance
- `confidence_score`: model confidence score
- `bbox_area_pixels`: detected object bounding-box size
- `lane_id`: lane where the vehicle was detected
- `estimated_speed_kmh`: estimated vehicle speed
- `is_emergency_vehicle`: emergency vehicle indicator

### `signal_performance_comparison.csv`

Per-second traffic signal performance log used for the A/B comparison.

Important columns:

- `second`: simulation second
- `controller_type`: controller group, either static or RL dynamic
- `active_phase`: active traffic signal phase
- `total_queue_cars`: total queued vehicles
- `avg_wait_time_sec`: average vehicle waiting time
- `vehicle_throughput_count`: vehicles cleared through the intersection
- `co2_emissions_kg`: estimated emissions
- `emergency_preemption_active`: emergency priority signal indicator

### `dashboard_summary.json`

Generated analytics payload consumed by the dashboard. It stores summary statistics, hypothesis-test outputs, impact metrics, time-series aggregates, phase distribution, vehicle mix, lane volume, and emergency vehicle counts.

### `traffic_signal_dashboard.html`

Interactive front-end dashboard built with HTML, CSS, JavaScript, and Chart.js. It fetches `dashboard_summary.json` and visualizes the project findings for a recruiter, interviewer, or stakeholder.

## Tech Stack

- Python
- Pandas
- NumPy
- SciPy
- HTML
- CSS
- JavaScript
- Chart.js
- JSON
- CSV-based analytics pipeline

## How To Run

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd traffic-rl-project
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Regenerate the analytics summary

```bash
python eda_traffic_analysis.py
```

This updates:

```text
dashboard_summary.json
```

### 4. Launch the dashboard

Because the dashboard loads `dashboard_summary.json`, the cleanest way to view it is through a local server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/traffic_signal_dashboard.html
```

## Methodology

The project treats the traffic-signal comparison as an A/B test:

- Group A: Static 120-second fixed-timer controller
- Group B: PPO-style reinforcement learning dynamic controller

For each controller, the analysis computes:

- Average wait time
- Queue length
- Vehicle throughput
- CO2 emissions
- Emergency preemption events
- Confidence intervals

Welch's t-test is used because it does not assume equal variance between the two groups. Cohen's d is used to measure effect size, which helps explain whether the improvement is practically meaningful in addition to being statistically significant.

## Business Impact

The analysis translates technical metrics into stakeholder-friendly impact:

- Lower waiting time means less congestion and better commuter experience.
- Higher throughput means more vehicles can pass through the intersection in the same time window.
- Lower queue length reduces road blockage and spillback risk.
- Lower idling time reduces CO2 emissions and fuel consumption.
- Emergency preemption support helps prioritize ambulances and other critical vehicles.

## What This Project Demonstrates

This project demonstrates the ability to:

- Build an end-to-end analytics pipeline from raw logs to executive dashboard
- Apply statistical testing to compare AI and baseline systems
- Use computer vision outputs as input for traffic intelligence
- Convert simulation data into practical business and environmental metrics
- Present technical results in a recruiter-friendly and stakeholder-friendly format
- Work across data analytics, machine learning evaluation, and front-end visualization

## Future Improvements

- Integrate live video-stream inference using a YOLO model
- Connect the RL controller to a traffic simulator such as SUMO
- Add real-time dashboard updates through a backend API
- Store experiment runs in a database for long-term comparison
- Add geospatial visualization for multi-intersection traffic networks
- Extend the controller to optimize for pedestrians, public transport, and emergency vehicles

## Resume Summary

AI-powered traffic signal analytics project comparing static fixed-time control with reinforcement-learning-based dynamic control. Built a Python analytics pipeline and Chart.js dashboard to evaluate congestion, throughput, emissions, and emergency vehicle behavior using traffic signal logs and computer vision detection data.

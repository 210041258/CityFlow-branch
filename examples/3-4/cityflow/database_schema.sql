-- CityFlow Simulation Database Schema
-- Compatible with PostgreSQL, MySQL, and SQLite
-- Supports multi-tenant simulation environment with real-time analytics

-- ===== SIMULATIONS TABLE =====
-- Stores simulation metadata and lifecycle information
CREATE TABLE IF NOT EXISTS simulations (
    id VARCHAR(36) PRIMARY KEY COMMENT 'Unique simulation ID (UUID or custom format)',
    name VARCHAR(255) NOT NULL COMMENT 'User-defined simulation name',
    config_path VARCHAR(512) NOT NULL COMMENT 'Path to CityFlow config JSON',
    roadnet_path VARCHAR(512) COMMENT 'Path to roadnet JSON file',
    flow_path VARCHAR(512) COMMENT 'Path to flow JSON file',
    algorithm VARCHAR(50) NOT NULL DEFAULT 'adaptive' COMMENT 'Traffic control algorithm: fixed-time, adaptive, optimized',
    status VARCHAR(50) NOT NULL DEFAULT 'initialized' COMMENT 'Status: initialized, running, paused, completed, error',
    step_count INT NOT NULL DEFAULT 0 COMMENT 'Current simulation step',
    total_time FLOAT DEFAULT 0.0 COMMENT 'Simulated time in seconds',
    max_steps INT COMMENT 'Maximum steps to run (null = infinite)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Simulation creation timestamp',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    started_at TIMESTAMP NULL COMMENT 'When simulation started running',
    completed_at TIMESTAMP NULL COMMENT 'When simulation completed',
    error_message TEXT COMMENT 'Error description if simulation failed',
    metadata JSON COMMENT 'Additional configuration/metadata',
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_algorithm (algorithm)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== SIMULATION METRICS TABLE =====
-- Records aggregated metrics at each simulation step
CREATE TABLE IF NOT EXISTS simulation_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL COMMENT 'Foreign key to simulations table',
    step INT NOT NULL COMMENT 'Simulation step number',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'When metric was recorded',
    simulated_time FLOAT DEFAULT 0.0 COMMENT 'Simulated time in seconds',
    total_vehicles INT DEFAULT 0 COMMENT 'Total active vehicles',
    avg_speed FLOAT DEFAULT 0.0 COMMENT 'Average vehicle speed (m/s)',
    max_speed FLOAT DEFAULT 0.0 COMMENT 'Maximum vehicle speed',
    min_speed FLOAT DEFAULT 0.0 COMMENT 'Minimum vehicle speed',
    vehicles_waiting INT DEFAULT 0 COMMENT 'Count of waiting/stopped vehicles',
    total_travel_distance FLOAT DEFAULT 0.0 COMMENT 'Sum of all vehicle travel distances',
    total_wait_time FLOAT DEFAULT 0.0 COMMENT 'Sum of all vehicle wait times',
    congestion_level FLOAT DEFAULT 0.0 COMMENT 'Normalized congestion (0.0-1.0)',
    congestion_state VARCHAR(20) COMMENT 'Discrete state: FREE, LIGHT, MODERATE, HEAVY, SEVERE',
    avg_queue_length FLOAT DEFAULT 0.0 COMMENT 'Average queue length across intersections',
    throughput INT DEFAULT 0 COMMENT 'Vehicles that completed route this step',
    emission_co2 FLOAT DEFAULT 0.0 COMMENT 'Estimated CO2 emissions (kg)',
    emission_nox FLOAT DEFAULT 0.0 COMMENT 'Estimated NOx emissions (kg)',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation_step (simulation_id, step),
    INDEX idx_timestamp (timestamp),
    INDEX idx_congestion_level (congestion_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== TRAFFIC CONTROL METRICS TABLE =====
-- Stores algorithm-specific decision metrics
CREATE TABLE IF NOT EXISTS algorithm_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    algorithm_type VARCHAR(50) NOT NULL COMMENT 'Algorithm type used',
    total_decisions INT DEFAULT 0 COMMENT 'Total signal phase changes',
    optimization_events INT DEFAULT 0 COMMENT 'Algorithm optimizations triggered',
    avg_phase_duration FLOAT DEFAULT 0.0 COMMENT 'Average traffic light phase duration',
    phase_changes INT DEFAULT 0 COMMENT 'Number of phase changes this step',
    decision_latency FLOAT DEFAULT 0.0 COMMENT 'Time to make decision (ms)',
    effectiveness_score FLOAT DEFAULT 0.0 COMMENT 'Algorithm effectiveness metric (0-1)',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation_algorithm (simulation_id, algorithm_type),
    INDEX idx_step (step)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== CONGESTION ANALYSIS TABLE =====
-- Stores congestion state transitions and analysis
CREATE TABLE IF NOT EXISTS congestion_analysis (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    congestion_value FLOAT DEFAULT 0.0 COMMENT 'Continuous congestion value (0-1)',
    congestion_state INT DEFAULT 0 COMMENT 'Discrete state (0=FREE, 1=LIGHT, 2=MODERATE, 3=HEAVY, 4=SEVERE)',
    state_transitions INT DEFAULT 0 COMMENT 'Cumulative state transitions',
    most_common_state INT COMMENT 'Most frequently observed state',
    state_duration_seconds INT DEFAULT 0 COMMENT 'How long in current state',
    affected_intersections INT DEFAULT 0 COMMENT 'Count of congested intersections',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation_congestion (simulation_id, congestion_state),
    INDEX idx_step (step)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== VEHICLES TABLE =====
-- Stores vehicle trajectory and state data
CREATE TABLE IF NOT EXISTS vehicles (
    id VARCHAR(36) PRIMARY KEY COMMENT 'Vehicle ID from simulation',
    simulation_id VARCHAR(36) NOT NULL,
    vehicle_type VARCHAR(50) DEFAULT 'car' COMMENT 'car, truck, bus, etc.',
    route_id VARCHAR(100) COMMENT 'Assigned route ID',
    source_junction VARCHAR(100) COMMENT 'Starting junction',
    destination_junction VARCHAR(100) COMMENT 'Target junction',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation (simulation_id),
    INDEX idx_vehicle_type (vehicle_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== VEHICLE TRAJECTORIES TABLE =====
-- High-frequency vehicle position and state updates
CREATE TABLE IF NOT EXISTS vehicle_trajectories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    vehicle_id VARCHAR(36) NOT NULL,
    step INT NOT NULL COMMENT 'Simulation step',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    x FLOAT DEFAULT 0.0 COMMENT 'X coordinate',
    y FLOAT DEFAULT 0.0 COMMENT 'Y coordinate',
    speed FLOAT DEFAULT 0.0 COMMENT 'Current speed (m/s)',
    acceleration FLOAT DEFAULT 0.0 COMMENT 'Current acceleration',
    direction FLOAT DEFAULT 0.0 COMMENT 'Direction in degrees (0-360)',
    road_id VARCHAR(100) COMMENT 'Current road segment',
    lane_id VARCHAR(100) COMMENT 'Current lane',
    is_waiting BOOLEAN DEFAULT FALSE COMMENT 'Vehicle is stopped/waiting',
    wait_time FLOAT DEFAULT 0.0 COMMENT 'Cumulative wait time (seconds)',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    INDEX idx_simulation_step (simulation_id, step),
    INDEX idx_vehicle_step (vehicle_id, step),
    INDEX idx_road (road_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== INTERSECTIONS TABLE =====
-- Stores intersection configuration and state
CREATE TABLE IF NOT EXISTS intersections (
    id VARCHAR(100) PRIMARY KEY COMMENT 'Intersection ID from roadnet',
    simulation_id VARCHAR(36),
    junction_name VARCHAR(255) COMMENT 'Human-readable name',
    junction_type VARCHAR(50) DEFAULT 'traffic_light' COMMENT 'traffic_light, roundabout, priority',
    x FLOAT DEFAULT 0.0 COMMENT 'X coordinate',
    y FLOAT DEFAULT 0.0 COMMENT 'Y coordinate',
    num_phases INT DEFAULT 2 COMMENT 'Number of traffic signal phases',
    cycle_time INT DEFAULT 60 COMMENT 'Full cycle time in seconds',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation (simulation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== SIGNAL STATE TABLE =====
-- Records traffic light signal state at each step
CREATE TABLE IF NOT EXISTS signal_states (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    intersection_id VARCHAR(100) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_phase INT DEFAULT 0 COMMENT 'Current phase index (0 or 1)',
    phase_name VARCHAR(50) COMMENT 'Phase name: NS, EW, etc.',
    phase_duration INT DEFAULT 30 COMMENT 'Current phase duration (seconds)',
    remaining_time INT DEFAULT 30 COMMENT 'Seconds remaining in phase',
    signal_state VARCHAR(20) COMMENT 'GREEN, YELLOW, RED',
    vehicles_ns INT DEFAULT 0 COMMENT 'Vehicles on North-South approaches',
    vehicles_ew INT DEFAULT 0 COMMENT 'Vehicles on East-West approaches',
    queue_length_ns INT DEFAULT 0 COMMENT 'Queue length NS',
    queue_length_ew INT DEFAULT 0 COMMENT 'Queue length EW',
    throughput_ns INT DEFAULT 0 COMMENT 'Vehicles passed NS',
    throughput_ew INT DEFAULT 0 COMMENT 'Vehicles passed EW',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    FOREIGN KEY (intersection_id) REFERENCES intersections(id) ON DELETE CASCADE,
    INDEX idx_simulation_intersection (simulation_id, intersection_id, step),
    INDEX idx_step (step)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== V2X COMMUNICATION TABLE =====
-- Records vehicle-to-infrastructure communication events
CREATE TABLE IF NOT EXISTS v2x_communications (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    vehicle_id VARCHAR(36) NOT NULL,
    intersection_id VARCHAR(100) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_type VARCHAR(50) COMMENT 'POSITION_UPDATE, QUEUE_LENGTH, EMERGENCY, etc.',
    data JSON COMMENT 'Message payload',
    latency_ms FLOAT DEFAULT 0.0 COMMENT 'Communication latency',
    success BOOLEAN DEFAULT TRUE COMMENT 'Message delivery success',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    FOREIGN KEY (intersection_id) REFERENCES intersections(id) ON DELETE CASCADE,
    INDEX idx_simulation_v2x (simulation_id, step),
    INDEX idx_vehicle_v2x (vehicle_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== V2V COMMUNICATION TABLE =====
-- Records vehicle-to-vehicle communication events
CREATE TABLE IF NOT EXISTS v2v_communications (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    vehicle_id_from VARCHAR(36) NOT NULL,
    vehicle_id_to VARCHAR(36) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_type VARCHAR(50) COMMENT 'COOPERATIVE_DRIVING, COLLISION_WARN, SPEED_SYNC, etc.',
    data JSON COMMENT 'Message payload',
    distance FLOAT DEFAULT 0.0 COMMENT 'Distance between vehicles (meters)',
    latency_ms FLOAT DEFAULT 0.0 COMMENT 'Communication latency',
    success BOOLEAN DEFAULT TRUE COMMENT 'Message delivery success',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation_v2v (simulation_id, step),
    INDEX idx_vehicle_from (vehicle_id_from),
    INDEX idx_distance (distance)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== ALERTS/ANOMALIES TABLE =====
-- Stores events, warnings, and anomalies detected during simulation
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    step INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(50) NOT NULL COMMENT 'CONGESTION, COLLISION, DEADLOCK, V2X_FAILURE, etc.',
    severity VARCHAR(20) NOT NULL COMMENT 'INFO, WARNING, CRITICAL',
    location_id VARCHAR(100) COMMENT 'Intersection or road ID affected',
    message TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP NULL,
    data JSON COMMENT 'Additional context',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation_alert (simulation_id, alert_type),
    INDEX idx_severity (severity),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== SIMULATION EXPORTS TABLE =====
-- Tracks data exports and result files
CREATE TABLE IF NOT EXISTS simulation_exports (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    export_type VARCHAR(50) COMMENT 'CSV, JSON, TRAJECTORY, HEATMAP, REPORT',
    export_path VARCHAR(512) NOT NULL COMMENT 'File path or URL',
    file_size BIGINT COMMENT 'File size in bytes',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exported_by VARCHAR(100) COMMENT 'User or service that triggered export',
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE,
    INDEX idx_simulation (simulation_id),
    INDEX idx_export_type (export_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== INDEXES FOR COMMON QUERIES =====
-- Performance optimization for frequently accessed patterns

-- Query: Get recent simulations
CREATE INDEX idx_simulations_recent ON simulations(created_at DESC, status);

-- Query: Get metrics for dashboard during specific time range
CREATE INDEX idx_metrics_time_range ON simulation_metrics(simulation_id, timestamp DESC);

-- Query: Get vehicle state at specific step
CREATE INDEX idx_trajectories_state ON vehicle_trajectories(simulation_id, step, vehicle_id);

-- Query: Analyze congestion patterns over time
CREATE INDEX idx_congestion_time ON congestion_analysis(simulation_id, step, congestion_state);

-- Query: Track intersection performance
CREATE INDEX idx_signals_performance ON signal_states(simulation_id, intersection_id, step);

-- ===== MATERIALIZED VIEWS / SUMMARIES (Optional) =====
-- For faster aggregated queries, consider creating summary tables

-- Hourly metrics summary (refresh periodically)
CREATE TABLE IF NOT EXISTS metrics_hourly_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(36) NOT NULL,
    hour_start TIMESTAMP,
    total_vehicles_peak INT,
    avg_congestion FLOAT,
    total_vehicles_completed INT,
    total_wait_time FLOAT,
    avg_speed FLOAT,
    throughput_efficiency FLOAT,
    UNIQUE KEY unique_sim_hour (simulation_id, hour_start),
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== STORED PROCEDURES (Optional, for MySQL) =====
-- Example: Insert metric and update simulation status in transaction

DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS record_simulation_metric(
    IN p_simulation_id VARCHAR(36),
    IN p_step INT,
    IN p_total_vehicles INT,
    IN p_avg_speed FLOAT,
    IN p_congestion_level FLOAT,
    IN p_vehicles_waiting INT
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO simulation_metrics (
        simulation_id, step, total_vehicles, avg_speed,
        congestion_level, vehicles_waiting
    ) VALUES (
        p_simulation_id, p_step, p_total_vehicles, p_avg_speed,
        p_congestion_level, p_vehicles_waiting
    );
    
    UPDATE simulations
    SET step_count = p_step,
        total_time = p_step * 0.1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_simulation_id;
    
    COMMIT;
END$$

DELIMITER ;

-- ===== GRANT PERMISSIONS (if using MySQL) =====
-- Create application user with appropriate privileges
-- GRANT SELECT, INSERT, UPDATE ON cityflow_db.* TO 'cityflow_app'@'%' IDENTIFIED BY 'strong_password';
-- GRANT SELECT ON cityflow_db.simulations TO 'cityflow_readonly'@'%' IDENTIFIED BY 'password';

"""
PostgreSQL Migration Script for CityFlow Simulation Database
Alembic-based migrations for version control and rollback support

Installation:
  pip install alembic sqlalchemy psycopg2-binary

Initialize Alembic:
  alembic init alembic
  
Configure in alembic.ini:
  sqlalchemy.url = postgresql://user:password@localhost/cityflow_db

Generate migration:
  alembic revision --autogenerate -m "description"

Apply migrations:
  alembic upgrade head

Rollback:
  alembic downgrade -1
"""

# File: alembic/versions/001_initial_schema.py

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    """Create initial CityFlow database schema"""
    
    # Create simulations table
    op.create_table(
        'simulations',
        sa.Column('id', sa.String(36), nullable=False, primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('config_path', sa.String(512), nullable=False),
        sa.Column('roadnet_path', sa.String(512), nullable=True),
        sa.Column('flow_path', sa.String(512), nullable=True),
        sa.Column('algorithm', sa.String(50), nullable=False, server_default='adaptive'),
        sa.Column('status', sa.String(50), nullable=False, server_default='initialized'),
        sa.Column('step_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_time', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('max_steps', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_status', 'simulations', ['status'])
    op.create_index('idx_created_at', 'simulations', ['created_at'])
    op.create_index('idx_algorithm', 'simulations', ['algorithm'])
    
    # Create simulation_metrics table
    op.create_table(
        'simulation_metrics',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('simulated_time', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('total_vehicles', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('avg_speed', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('max_speed', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('min_speed', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('vehicles_waiting', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_travel_distance', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('total_wait_time', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('congestion_level', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('congestion_state', sa.String(20), nullable=True),
        sa.Column('avg_queue_length', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('throughput', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('emission_co2', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('emission_nox', sa.Float(), nullable=True, server_default='0.0'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_simulation_step', 'simulation_metrics', ['simulation_id', 'step'])
    op.create_index('idx_metrics_timestamp', 'simulation_metrics', ['timestamp'])
    op.create_index('idx_congestion_level', 'simulation_metrics', ['congestion_level'])
    
    # Create algorithm_metrics table
    op.create_table(
        'algorithm_metrics',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('algorithm_type', sa.String(50), nullable=False),
        sa.Column('total_decisions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('optimization_events', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('avg_phase_duration', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('phase_changes', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('decision_latency', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('effectiveness_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_simulation_algorithm', 'algorithm_metrics', ['simulation_id', 'algorithm_type'])
    op.create_index('idx_algo_step', 'algorithm_metrics', ['step'])
    
    # Create congestion_analysis table
    op.create_table(
        'congestion_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('congestion_value', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('congestion_state', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('state_transitions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('most_common_state', sa.Integer(), nullable=True),
        sa.Column('state_duration_seconds', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('affected_intersections', sa.Integer(), nullable=True, server_default='0'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_simulation_congestion', 'congestion_analysis', ['simulation_id', 'congestion_state'])
    op.create_index('idx_cong_step', 'congestion_analysis', ['step'])
    
    # Create vehicles table
    op.create_table(
        'vehicles',
        sa.Column('id', sa.String(36), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('vehicle_type', sa.String(50), nullable=True, server_default='car'),
        sa.Column('route_id', sa.String(100), nullable=True),
        sa.Column('source_junction', sa.String(100), nullable=True),
        sa.Column('destination_junction', sa.String(100), nullable=True),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_vehicles_simulation', 'vehicles', ['simulation_id'])
    op.create_index('idx_vehicle_type', 'vehicles', ['vehicle_type'])
    
    # Create vehicle_trajectories table (high-frequency updates)
    op.create_table(
        'vehicle_trajectories',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('vehicle_id', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('x', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('y', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('speed', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('acceleration', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('direction', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('road_id', sa.String(100), nullable=True),
        sa.Column('lane_id', sa.String(100), nullable=True),
        sa.Column('is_waiting', sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column('wait_time', sa.Float(), nullable=True, server_default='0.0'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['vehicle_id'], ['vehicles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_simulation_step_traj', 'vehicle_trajectories', ['simulation_id', 'step'])
    op.create_index('idx_vehicle_step', 'vehicle_trajectories', ['vehicle_id', 'step'])
    op.create_index('idx_trajectory_road', 'vehicle_trajectories', ['road_id'])
    op.create_index('idx_trajectory_timestamp', 'vehicle_trajectories', ['timestamp'])
    
    # Create intersections table
    op.create_table(
        'intersections',
        sa.Column('id', sa.String(100), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=True),
        sa.Column('junction_name', sa.String(255), nullable=True),
        sa.Column('junction_type', sa.String(50), nullable=True, server_default='traffic_light'),
        sa.Column('x', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('y', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('num_phases', sa.Integer(), nullable=True, server_default='2'),
        sa.Column('cycle_time', sa.Integer(), nullable=True, server_default='60'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_intersections_simulation', 'intersections', ['simulation_id'])
    
    # Create signal_states table
    op.create_table(
        'signal_states',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('intersection_id', sa.String(100), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('current_phase', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('phase_name', sa.String(50), nullable=True),
        sa.Column('phase_duration', sa.Integer(), nullable=True, server_default='30'),
        sa.Column('remaining_time', sa.Integer(), nullable=True, server_default='30'),
        sa.Column('signal_state', sa.String(20), nullable=True),
        sa.Column('vehicles_ns', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('vehicles_ew', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('queue_length_ns', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('queue_length_ew', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('throughput_ns', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('throughput_ew', sa.Integer(), nullable=True, server_default='0'),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['intersection_id'], ['intersections.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_simulation_intersection_step', 'signal_states', ['simulation_id', 'intersection_id', 'step'])
    op.create_index('idx_signal_step', 'signal_states', ['step'])
    
    # Create v2x_communications table
    op.create_table(
        'v2x_communications',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('vehicle_id', sa.String(36), nullable=False),
        sa.Column('intersection_id', sa.String(100), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('message_type', sa.String(50), nullable=True),
        sa.Column('data', postgresql.JSON(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('success', sa.Boolean(), nullable=True, server_default=sa.true()),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['vehicle_id'], ['vehicles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['intersection_id'], ['intersections.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_v2x_simulation_step', 'v2x_communications', ['simulation_id', 'step'])
    op.create_index('idx_v2x_vehicle', 'v2x_communications', ['vehicle_id'])
    
    # Create v2v_communications table
    op.create_table(
        'v2v_communications',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('vehicle_id_from', sa.String(36), nullable=False),
        sa.Column('vehicle_id_to', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('message_type', sa.String(50), nullable=True),
        sa.Column('data', postgresql.JSON(), nullable=True),
        sa.Column('distance', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('latency_ms', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('success', sa.Boolean(), nullable=True, server_default=sa.true()),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_v2v_simulation_step', 'v2v_communications', ['simulation_id', 'step'])
    op.create_index('idx_v2v_vehicle_from', 'v2v_communications', ['vehicle_id_from'])
    op.create_index('idx_v2v_distance', 'v2v_communications', ['distance'])
    
    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('location_id', sa.String(100), nullable=True),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('resolved', sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('data', postgresql.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alerts_simulation_type', 'alerts', ['simulation_id', 'alert_type'])
    op.create_index('idx_alerts_severity', 'alerts', ['severity'])
    op.create_index('idx_alerts_timestamp', 'alerts', ['timestamp'])
    
    # Create simulation_exports table
    op.create_table(
        'simulation_exports',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('export_type', sa.String(50), nullable=True),
        sa.Column('export_path', sa.String(512), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('exported_by', sa.String(100), nullable=True),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_exports_simulation', 'simulation_exports', ['simulation_id'])
    op.create_index('idx_exports_type', 'simulation_exports', ['export_type'])
    
    # Create metrics_hourly_summary table (for analytics dashboards)
    op.create_table(
        'metrics_hourly_summary',
        sa.Column('id', sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column('simulation_id', sa.String(36), nullable=False),
        sa.Column('hour_start', sa.DateTime(), nullable=True),
        sa.Column('total_vehicles_peak', sa.Integer(), nullable=True),
        sa.Column('avg_congestion', sa.Float(), nullable=True),
        sa.Column('total_vehicles_completed', sa.Integer(), nullable=True),
        sa.Column('total_wait_time', sa.Float(), nullable=True),
        sa.Column('avg_speed', sa.Float(), nullable=True),
        sa.Column('throughput_efficiency', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['simulation_id'], ['simulations.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('simulation_id', 'hour_start', name='unique_sim_hour'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    """Rollback: Drop all tables in reverse order"""
    op.drop_table('metrics_hourly_summary')
    op.drop_table('simulation_exports')
    op.drop_table('alerts')
    op.drop_table('v2v_communications')
    op.drop_table('v2x_communications')
    op.drop_table('signal_states')
    op.drop_table('intersections')
    op.drop_table('vehicle_trajectories')
    op.drop_table('vehicles')
    op.drop_table('congestion_analysis')
    op.drop_table('algorithm_metrics')
    op.drop_table('simulation_metrics')
    op.drop_table('simulations')

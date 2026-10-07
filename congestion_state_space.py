#!/usr/bin/env python3
"""
Feature Counter Module - Congestion State Space
Used by: run.py --training-env (for congestion discretization)
"""

from collections import defaultdict

class CongestionStateSpace:
    """Discretized congestion state space for training"""
    
    def __init__(self, levels=5):  # 5 discrete congestion levels: FREE, LIGHT, MODERATE, HEAVY, SEVERE
        self.levels = levels
        self.level_names = ['FREE', 'LIGHT', 'MODERATE', 'HEAVY', 'SEVERE']
        self.congestion_levels_history = []
        self.state_transitions = defaultdict(int)
        self.current_state = 0
        
    def discretize_congestion(self, congestion_value):
        """Convert continuous congestion (0-1) to discrete state (0-levels-1)"""
        if congestion_value < 0.2:
            return 0  # FREE
        elif congestion_value < 0.4:
            return 1  # LIGHT
        elif congestion_value < 0.6:
            return 2  # MODERATE
        elif congestion_value < 0.8:
            return 3  # HEAVY
        else:
            return 4  # SEVERE
    
    def update_state(self, congestion_value):
        """Update congestion state and track transitions"""
        new_state = self.discretize_congestion(congestion_value)
        self.congestion_levels_history.append(new_state)
        
        if self.congestion_levels_history:
            transition = (self.current_state, new_state)
            self.state_transitions[transition] += 1
        
        self.current_state = new_state
        return new_state
    
    def get_state_name(self, state):
        """Get human-readable name for state"""
        return self.level_names[state] if 0 <= state < len(self.level_names) else 'UNKNOWN'
    
    def get_metrics(self):
        """Get congestion state space metrics"""
        state_counts = defaultdict(int)
        for state in self.congestion_levels_history:
            state_counts[state] += 1
        
        return {
            'congestion_state_space_levels': self.levels,
            'congestion_total_state_transitions': len(self.state_transitions),
            'congestion_current_state': self.current_state,
            'congestion_current_state_name': self.get_state_name(self.current_state),
            'congestion_state_distribution': dict(state_counts),
            'congestion_most_common_state': max(state_counts, key=state_counts.get) if state_counts else 0
        }


if __name__ == '__main__':
    # Example usage
    cong_space = CongestionStateSpace()
    for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        state = cong_space.update_state(val)
        print(f"Congestion: {val} -> State: {cong_space.get_state_name(state)}")
    print("\nMetrics:", cong_space.get_metrics())

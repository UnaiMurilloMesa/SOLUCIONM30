import pandas as pd
import numpy as np
import math
from .physics import TrafficPhysics

class TrafficOptimizer:
    """
    Optimizer engine for Variable Speed Limits (VSL).
    Implements advanced logic to finding optimal limit [10,20...90] 
    that maximizes flow/harmonization given current conditions.
    """
    
    def __init__(self, critical_density_override: float = None, 
                 max_capacity_override: float = None, 
                 base_speed_limit: int = 90):
        self.critical_density = critical_density_override
        self.max_capacity = max_capacity_override
        self.base_speed_limit = base_speed_limit

    def _round_speed(self, speed: float) -> int:
        return int(round(speed / 10.0) * 10)

    def optimize_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        df_opt = df.copy()
        
        # 1. Determine Parameters
        k_crit = self.critical_density or TrafficPhysics.calculate_critical_density(df_opt)
        q_max = self.max_capacity or TrafficPhysics.calculate_max_capacity(df_opt)
            
        print(f"   [Optimizer] Advanced VSL Logic. K_crit={k_crit:.2f}, Q_max={q_max:.0f}, Base={self.base_speed_limit}")
        
        # Candidates limits
        # [10, 20, ..., 90] but <= base_speed_limit
        max_limit_int = int(self.base_speed_limit)
        limit_candidates = [l for l in range(10, 100, 10) if l <= max_limit_int]
        # Ensure base limit is included if not multiple of 10
        if max_limit_int not in limit_candidates:
            limit_candidates.append(max_limit_int)
        limit_candidates.sort()

        # 2. Define Advanced Simulation Function
        def simular_escenario_avanzado(row):
            densidad_real = row.get('density', 0)
            flujo_real = row.get('intensidad', 0)
            velocidad_real = row.get('vmed', 0)
            
            # Constraint: Free Flow (Low Density).
            # If density < 0.8 * k_crit, we assume no intervention needed (Limit = Base).
            # BUT: Check for "Ghost Jam" (Low Density but Low Speed).
            # If speed is < 40 km/h and limit is > 50, we should probably intervene even if density says low.
            is_low_density = densidad_real < (k_crit * 0.8)
            is_normal_speed = velocidad_real > 40
            
            if is_low_density and is_normal_speed:
                return flujo_real, velocidad_real, self.base_speed_limit

            # --- OPTIMIZATION SEARCH ---
            best_flow = -1
            best_speed = -1
            best_limit = self.base_speed_limit
            
            # Compliance Params
            compliance_rate = 0.8 # 20% speeding
            max_improvement = 0.15 # 15% theoretical max gain
            
            for limit in limit_candidates:
                # MODEL:
                # 1. Harmonization Bonus:
                # High bonus if Limit is close to Real Speed (slightly above).
                # Example: real=20, limit=30 -> Great.
                # real=20, limit=90 -> Bad (variance high).
                # real=20, limit=10 -> Bad (forced slowdown).
                
                # Gap calculation
                gap = limit - velocidad_real
                
                # "Alpha" factor (Recovery Factor)
                # We model this as a peak around +10 km/h gap.
                # If limit == v_real + 10 => Max Bonus.
                # If limit >> v_real => Bonus decays to 0.
                # If limit < v_real => Penalty or no bonus, but also PHYSICALLY constrained speed.
                
                alpha = 0.0
                
                if gap >= -5: # Allow slightly below (forcing discipline) to +Infinity
                    # Gaussian-like decay
                    # Center at 10km/h
                    # Sigma = 15km/h width
                    distance_from_sweet_spot = abs(gap - 10)
                    # Gaussian: exp( - (x^2) / (2*sigma^2) )
                    alpha = max_improvement * math.exp( - (distance_from_sweet_spot**2) / (2 * (15**2)) )
                else:
                    # Limit is WAY below speed (e.g. going 50, limit 20).
                    # This causes braking shockwave. Negative impact?
                    # Or simply capacity drop (alpha = 0) and speed is capped.
                    alpha = 0.0
                
                # Apply Compliance to Alpha
                real_alpha_improvement = alpha * compliance_rate
                
                # Predict Flow
                # Q_new = Q_real * (1 + alpha)
                predicted_flow = min(flujo_real * (1 + real_alpha_improvement), q_max)
                
                # Predict Speed
                # V = Q / K
                if densidad_real > 0:
                    predicted_speed = predicted_flow / densidad_real
                else:
                    predicted_speed = limit
                
                # Constraints
                # 1. Physics: Speed cannot exceed the Limit significantly (unless non-compliant).
                # But we want the "Resulting Speed" of the system.
                # If limit is 20, speed -> approx 20 (or slightly higher due to 20% compliance).
                # Let's say effective_limit = limit * compliance + (limit+20)*non_compliance?
                # Simplify: V_pred = min(calculated_from_flow, limit * 1.0) 
                # Actually, V_pred based on flow MIGHT imply a speed > limit if density dropped.
                # But here density is constant.
                # So V_pred from Q/K is the 'Harmonized Speed'.
                # But we must cap it at the limit we set, because that's the constraint.
                # Users won't drive 50 if limit is 30 in steady state (mostly).
                
                # We cap at MAX(limit, current_speed) ?
                # If we force limit 20, and current is 20, and flow improves -> V becomes 22 -> OK cap at 20? 
                # No, if flow improves, speed improves.
                
                # If we set limit 30, and Q/K says 25. Speed is 25. Max is 30. OK.
                # If we set limit 10. Q/K says 25? Impossible. Limit constrains flow to Q = K*10.
                # So check consistency:
                phys_max_speed = limit
                
                # If calculated flow implies speed > limit, we are inconsistent.
                # Flow is constrained by Limit * K.
                # So actual Q_pred = min(Q_pred, limit * K)
                
                if predicted_speed > phys_max_speed:
                    predicted_speed = phys_max_speed
                    predicted_flow = predicted_speed * densidad_real
                
                # Selection Criteria
                # We want to MAXIMIZE FLOW.
                # But also ensure Speed isn't drastically killed (e.g. Limit 10 gives stability but Q is low).
                
                if predicted_flow > best_flow:
                    best_flow = predicted_flow
                    best_speed = predicted_speed
                    best_limit = limit
                elif predicted_flow == best_flow:
                    # Tie-breaker: Higher Speed
                    if predicted_speed > best_speed:
                        best_speed = predicted_speed
                        best_limit = limit
            
            # Final check: Don't produce a result worse than reality
            final_flow = max(best_flow, flujo_real)
            final_speed = final_flow / densidad_real if densidad_real > 0 else velocidad_real
            
            # Constraint: Speed must be <= base_limit
            final_speed = min(final_speed, self.base_speed_limit)
            
            return final_flow, final_speed, best_limit

        # 3. Apply
        result = df_opt.apply(simular_escenario_avanzado, axis=1, result_type='expand')
        df_opt[['intensidad_opt', 'velocidad_opt', 'limite_dinamico']] = result
        
        # 4. Rounding
        df_opt['velocidad_opt'] = df_opt['velocidad_opt'].apply(self._round_speed)
        
        # Aliases
        df_opt['simulated_speed'] = df_opt['velocidad_opt']
        df_opt['optimal_speed_limit'] = df_opt['limite_dinamico']

        return df_opt

"""Simplified level-track constant-deceleration regenerative braking model."""
import numpy as np
import pandas as pd

def simulate(mass_t=200, speed_kmh=60, decel=0.9, efficiency=0.8, max_power_kw=1500, steps=1001):
    if not (mass_t > 0 and speed_kmh > 0 and decel > 0 and 0 <= efficiency <= 1 and max_power_kw >= 0 and steps >= 2):
        raise ValueError('Invalid simulation inputs')
    m=mass_t*1000; v0=speed_kmh/3.6; duration=v0/decel
    t=np.linspace(0,duration,steps); v=np.maximum(v0-decel*t,0)
    distance=v0*t-0.5*decel*t*t
    mechanical=m*decel*v
    ideal_electrical=mechanical*efficiency
    recovered=np.minimum(ideal_electrical,max_power_kw*1000)
    conversion_loss=mechanical-ideal_electrical
    curtailed=ideal_electrical-recovered
    def integrate(y): return float(np.trapezoid(y,t))
    initial=0.5*m*v0*v0
    rec=integrate(recovered); loss=integrate(conversion_loss); curtailed_j=integrate(curtailed)
    # Constant deceleration demands mechanical braking regardless of electrical acceptance.
    df=pd.DataFrame({'time_s':t,'speed_kmh':v*3.6,'distance_m':distance,'mechanical_power_kw':mechanical/1000,'recovered_power_kw':recovered/1000,'conversion_loss_kw':conversion_loss/1000,'unaccepted_power_kw':curtailed/1000,'remaining_kinetic_kwh':0.5*m*v*v/3.6e6})
    summary={'stop_time_s':duration,'stop_distance_m':v0*v0/(2*decel),'initial_kwh':initial/3.6e6,'recovered_kwh':rec/3.6e6,'conversion_loss_kwh':loss/3.6e6,'unaccepted_kwh':curtailed_j/3.6e6,'not_recovered_kwh':(initial-rec)/3.6e6,'recovery_pct':100*rec/initial,'peak_mechanical_kw':mechanical[0]/1000,'peak_recovered_kw':float(recovered.max()/1000)}
    return summary,df

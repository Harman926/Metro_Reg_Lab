import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import simulate
import json, os
from nn_model import predict as nn_predict, predict_df as nn_predict_df, model_available

st.set_page_config(page_title='Metro Regen Lab',page_icon='🚇',layout='wide')
st.title('🚇 Metro Regen Lab')
st.caption('Design and Simulation of a Regenerative Braking System for an Electric Metro Train | UEE Group Project')
with st.sidebar:
    st.header('Train parameters')
    mass=st.slider('Train mass (tonnes)',80,400,200,10)
    speed=st.slider('Approach speed (km/h)',20,120,60,5)
    decel=st.slider('Deceleration (m/s²)',0.3,1.5,0.9,0.05)
    efficiency=st.slider('Regeneration conversion efficiency (%)',40,95,80,5)/100
    cap=st.slider('Maximum accepted electrical power (kW)',0,5000,1500,100)
    st.info('Illustrative parameters, not a calibrated model of any specific metro rolling stock.')
s,d=simulate(mass,speed,decel,efficiency,cap)
tabs=st.tabs(['Live braking simulation','Energy analysis','Scenario comparison','Model & validation','NN vs Physics'])
with tabs[0]:
    a,b,c,dcol=st.columns(4)
    a.metric('Stopping time',f"{s['stop_time_s']:.2f} s")
    b.metric('Stopping distance',f"{s['stop_distance_m']:.2f} m")
    c.metric('Recovered electricity',f"{s['recovered_kwh']:.2f} kWh")
    dcol.metric('Recovery fraction',f"{s['recovery_pct']:.1f}%")
    st.subheader('Train approaching the station')
    progress=st.slider('Braking animation / time position (%)',0,100,0,1)
    idx=round(progress/100*(len(d)-1)); state=d.iloc[idx]
    st.progress(progress/100)
    st.markdown(f"**🚇 Train** {'━'*max(1,int((100-progress)/7))} **🚉 Station**")
    st.write(f"Elapsed: **{state.time_s:.1f} s** | Speed: **{state.speed_kmh:.1f} km/h** | Distance travelled: **{state.distance_m:.1f} m** | Distance remaining: **{max(0,s['stop_distance_m']-state.distance_m):.1f} m**")
    fig,axs=plt.subplots(1,2,figsize=(11,3.5))
    axs[0].plot(d.time_s,d.speed_kmh);axs[0].scatter([state.time_s],[state.speed_kmh]);axs[0].set(xlabel='Time (s)',ylabel='Speed (km/h)',title='Speed during braking');axs[0].grid(alpha=.3)
    axs[1].plot(d.time_s,d.mechanical_power_kw,label='Mechanical braking');axs[1].plot(d.time_s,d.recovered_power_kw,label='Electrical recovery');axs[1].set(xlabel='Time (s)',ylabel='Power (kW)',title='Braking power');axs[1].legend();axs[1].grid(alpha=.3)
    st.pyplot(fig);plt.close(fig)
with tabs[1]:
    st.subheader('Where does the train’s kinetic energy go?')
    st.markdown('**Train kinetic energy → traction motor as generator → power converter → electrical supply / storage**')
    x,y,z=st.columns(3)
    x.metric('Initial kinetic energy',f"{s['initial_kwh']:.3f} kWh")
    y.metric('Recovered electricity',f"{s['recovered_kwh']:.3f} kWh")
    z.metric('Not recovered',f"{s['not_recovered_kwh']:.3f} kWh")
    fig,ax=plt.subplots(figsize=(9,2.8))
    labels=['Recovered electricity','Conversion losses','Unaccepted regeneration']
    vals=[s['recovered_kwh'],s['conversion_loss_kwh'],s['unaccepted_kwh']]
    ax.barh(['Initial kinetic energy'],[vals[0]],label=labels[0]);ax.barh(['Initial kinetic energy'],[vals[1]],left=vals[0],label=labels[1]);ax.barh(['Initial kinetic energy'],[vals[2]],left=vals[0]+vals[1],label=labels[2]);ax.set_xlabel('Energy (kWh)');ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3);st.pyplot(fig);plt.close(fig)
    st.write('**Conventional friction braking:** 0 kWh electrically recovered. **Regenerative braking:**',f"{s['recovered_kwh']:.3f} kWh recovered for this one stop.")
    st.warning('Unaccepted regeneration is energy that must be dissipated or handled by another braking path in this simplified model. Conversion losses are not identical to friction-brake heat. No claim is made that all recovered energy is subsequently used.')
with tabs[2]:
    st.subheader('Sensitivity analysis: change one variable at a time')
    mode=st.selectbox('Compare by',['Approach speed','Train mass','Efficiency','Accepted power limit'])
    settings={'Approach speed':('Speed (km/h)',np.arange(20,121,10)),'Train mass':('Mass (tonnes)',np.arange(80,401,40)),'Efficiency':('Efficiency (%)',np.arange(40,96,5)),'Accepted power limit':('Power limit (kW)',np.arange(0,5001,250))}
    label,values=settings[mode];records=[]
    for val in values:
        args=[mass,speed,decel,efficiency,cap]
        args[{'Approach speed':1,'Train mass':0,'Efficiency':3,'Accepted power limit':4}[mode]]=float(val)/100 if mode=='Efficiency' else float(val)
        res,_=simulate(*args)
        records.append({label:val,'Recovered energy (kWh)':res['recovered_kwh'],'Initial kinetic energy (kWh)':res['initial_kwh'],'Recovery (%)':res['recovery_pct']})
    comparison=pd.DataFrame(records)
    fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(comparison[label],comparison['Recovered energy (kWh)'],marker='o',label='Recovered energy');ax.plot(comparison[label],comparison['Initial kinetic energy (kWh)'],linestyle='--',label='Initial kinetic energy');ax.set(xlabel=label,ylabel='Energy (kWh)');ax.legend();ax.grid(alpha=.3);st.pyplot(fig);plt.close(fig)
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    st.download_button('Download comparison CSV',comparison.to_csv(index=False),'comparison.csv','text/csv')
with tabs[3]:
    st.subheader('Equations and assumptions')
    st.latex(r'E_k=\frac12mv_0^2\qquad t_s=\frac{v_0}{a}\qquad d_s=\frac{v_0^2}{2a}')
    st.latex(r'v(t)=v_0-at\qquad P_m(t)=mav(t)\qquad P_e(t)=\min(\eta P_m(t),P_{max})')
    st.latex(r'E_{recovered}=\int_0^{t_s}P_e(t)\,dt')
    st.write('Level track; constant deceleration; equivalent translational mass; no aerodynamic or rolling losses; no auxiliary consumption; fixed conversion efficiency; accepted electrical power capped; no low-speed regeneration cutout. Braking effort is assumed to be supplemented as needed to maintain the specified deceleration.')
    st.write('**Energy conservation check:**',f"Initial = {s['initial_kwh']:.5f} kWh; recovered + conversion loss + unaccepted = {s['recovered_kwh']+s['conversion_loss_kwh']+s['unaccepted_kwh']:.5f} kWh")
with tabs[4]:
    st.subheader('Neural-network surrogate vs physics model')
    st.caption('MLP (5 inputs → 64 → 64 → 32 → recovered kWh, recovery %) trained on simulate() outputs. Uses the sidebar parameters.')
    if not model_available():
        st.warning('Trained model not found. Run `python generate_dataset.py` then `python train_nn.py`.')
    else:
        variant=st.radio('NN variant',['direct','fraction'],horizontal=True,help='direct: MLP predicts kWh and %. fraction: MLP predicts % only; kWh = % × analytic kinetic energy.')
        nn=nn_predict(mass,speed,decel,efficiency,cap,variant)
        c1,c2,c3=st.columns(3)
        c1.metric('Physics recovered',f"{s['recovered_kwh']:.3f} kWh",f"{s['recovery_pct']:.2f}%",delta_color='off')
        c2.metric('NN recovered',f"{nn['recovered_kwh']:.3f} kWh",f"{nn['recovery_pct']:.2f}%",delta_color='off')
        c3.metric('Error (NN − physics)',f"{nn['recovered_kwh']-s['recovered_kwh']:+.3f} kWh",f"{nn['recovery_pct']-s['recovery_pct']:+.2f} pts",delta_color='off')
        if mass>400 or speed>120 or cap>5000:
            st.info('Inputs are outside the NN training range; expect larger errors.')
        st.markdown('**Power-limit sweep (other sliders fixed)**')
        caps=np.linspace(0,5000,101)
        phys=[simulate(mass,speed,decel,efficiency,float(c))[0]['recovery_pct'] for c in caps]
        sweep=pd.DataFrame({'mass_t':mass,'speed_kmh':speed,'decel':decel,'efficiency':efficiency,'max_power_kw':caps})
        pred=nn_predict_df(sweep,variant)['recovery_pct']
        fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(caps,phys,label='Physics');ax.plot(caps,pred,'--',label='NN');ax.axvline(cap,color='gray',ls=':',label='current limit');ax.set(xlabel='Accepted power limit (kW)',ylabel='Recovery (%)');ax.legend();ax.grid(alpha=.3);st.pyplot(fig);plt.close(fig)
        if os.path.exists('models/metrics.json'):
            st.markdown('**Held-out accuracy (from evaluate.py)**')
            m=json.load(open('models/metrics.json'));rows=[]
            for sname,label in [('test','In-range test'),('extrapolation','Out-of-range')]:
                for name,v in m[sname].items():
                    rows.append({'Set':label,'Model':name,'R² (kWh)':v['recovered_kwh']['R2'],'MAE (kWh)':v['recovered_kwh']['MAE'],'R² (%)':v['recovery_pct']['R2'],'MAE (% pts)':v['recovery_pct']['MAE']})
            st.dataframe(pd.DataFrame(rows).round(4),use_container_width=True,hide_index=True)
        for f,cap_txt in [('figures/parity_test.png','Parity plot (test set)'),('figures/residual_vs_cap_ratio.png','Residuals vs power-cap ratio')]:
            if os.path.exists(f): st.image(f,caption=cap_txt)
st.download_button('Download full time-series CSV',d.to_csv(index=False),'metro_braking_timeseries.csv','text/csv')

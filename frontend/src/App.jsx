import { useState } from 'react';
import './index.css';
import Dashboard from './Dashboard';

const STEPS = ['Demography', 'Tumor', 'Biomarkers', 'Treatment', 'Follow-up'];

const initialForm = {
  Age: 50, Sex: 'Female', Athleticity: 'Medium', BMI: 25.0,
  Smoking_Alcohol_History: 'None', Cancer_Type: 'Breast',
  Tumor_Stage: 2, Tumor_Grade: 1, Tumor_Size_cm: 2.5,
  Lymph_Nodes_Involved: 'No', Metastasis: 'No', Tumor_Type: 'Malignant',
  Hormone_Receptor: 'Not Applicable', Gene_Mutations: 'None',
  Surgery_Type: 'None', Chemotherapy: 'No', Radiation_Therapy: 'No',
  Hormone_Therapy: 'No', Immunotherapy: 'No',
  Time_Since_Treatment_Months: 12, Follow_Up_Visits: 3,
  Previous_Reoccurrence: 'No'
};

const surgeryOptions = {
  Breast: ['None', 'Lumpectomy', 'Mastectomy'],
  Lung: ['None', 'Lobectomy'],
  Colon: ['None', 'Resection'],
  Prostate: ['None', 'Prostatectomy'],
  Liver: ['None', 'Excision'],
  Mouth: ['None', 'Excision'],
  Thyroid: ['None', 'Excision', 'Thyroidectomy'],
};

function Field({ label, name, type = 'text', value, onChange, children, ...props }) {
  return (
    <div className="form-group">
      <label>{label}</label>
      {children || <input type={type} name={name} className="form-control" value={value} onChange={onChange} {...props} />}
    </div>
  );
}

function Select({ label, name, options, value, onChange }) {
  return (
    <Field label={label} name={name}>
      <select name={name} className="form-control" value={value} onChange={onChange}>
        {options.map(o => {
          const [val, lbl] = Array.isArray(o) ? o : [o, o];
          return <option key={val} value={val}>{lbl}</option>;
        })}
      </select>
    </Field>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const set = (name, value) => setForm(p => ({ ...p, [name]: value }));
  const handleChange = e => {
    const { name, value } = e.target;
    // Keep values as strings while typing to avoid breaking decimals (e.g., '2.')
    // FastAPI/Pydantic will automatically coerce them to int/float on the backend.
    set(name, value);
  };

  const next = () => setStep(s => Math.min(s + 1, STEPS.length - 1));
  const prev = () => setStep(s => Math.max(s - 1, 0));

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      // Coerce numeric fields before sending
      const payload = { ...form };
      ['Age', 'Tumor_Stage', 'Tumor_Grade', 'Time_Since_Treatment_Months', 'Follow_Up_Visits'].forEach(k => {
        if (payload[k] !== '') payload[k] = parseInt(payload[k], 10);
      });
      ['BMI', 'Tumor_Size_cm'].forEach(k => {
        if (payload[k] !== '') payload[k] = parseFloat(payload[k]);
      });

      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setResult(null); setError(null); setStep(0); setForm(initialForm); };

  // Gauge math
  const gaugeProps = (prob) => {
    const r = 90, c = 2 * Math.PI * r;
    const offset = c - (prob / 100) * c;
    const level = prob > 60 ? 'high' : prob > 35 ? 'medium' : 'low';
    return { r, c, offset, level };
  };

  // Step content
  const stepContent = [
    // 0: Demography
    <>
      <div className="form-section-title">
        <span className="icon blue"><Icon d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></span>
        Demography
      </div>
      <div className="form-grid">
        <Field label="Age" name="Age" type="number" min={18} max={100} required value={form.Age} onChange={handleChange} />
        <Select label="Sex" name="Sex" options={['Female', 'Male']} value={form.Sex} onChange={handleChange} />
        <Select label="Athleticity Level" name="Athleticity" options={['Low', 'Medium', 'High']} value={form.Athleticity} onChange={handleChange} />
        <Field label="BMI" name="BMI" type="number" step={0.1} min={10} max={60} required value={form.BMI} onChange={handleChange} />
        <Select label="Smoking / Alcohol History" name="Smoking_Alcohol_History"
          options={['None', 'Occasional', 'Frequent', 'Heavy']} value={form.Smoking_Alcohol_History} onChange={handleChange} />
      </div>
    </>,
    // 1: Tumor
    <>
      <div className="form-section-title">
        <span className="icon purple"><Icon d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></span>
        Tumor Characteristics
      </div>
      <div className="form-grid">
        <Select label="Cancer Type (Location)" name="Cancer_Type"
          options={['Breast', 'Lung', 'Colon', 'Prostate', 'Liver', 'Mouth', 'Thyroid']} value={form.Cancer_Type} onChange={handleChange} />
        <Field label="Tumor Stage (1–4)" name="Tumor_Stage" type="number" min={1} max={4} required value={form.Tumor_Stage} onChange={handleChange} />
        <Field label="Tumor Grade (1–3)" name="Tumor_Grade" type="number" min={1} max={3} required value={form.Tumor_Grade} onChange={handleChange} />
        <Field label="Tumor Size (cm)" name="Tumor_Size_cm" type="number" step={0.1} min={0.1} max={30} required value={form.Tumor_Size_cm} onChange={handleChange} />
        <Select label="Lymph Nodes Involved" name="Lymph_Nodes_Involved" options={['No', 'Yes']} value={form.Lymph_Nodes_Involved} onChange={handleChange} />
        <Select label="Metastasis Status" name="Metastasis" options={['No', 'Yes']} value={form.Metastasis} onChange={handleChange} />
        <Select label="Tumor Type" name="Tumor_Type" options={['Malignant', 'Benign']} value={form.Tumor_Type} onChange={handleChange} />
      </div>
    </>,
    // 2: Biomarkers
    <>
      <div className="form-section-title">
        <span className="icon cyan"><Icon d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></span>
        Biomarkers
      </div>
      <div className="form-grid">
        <Select label="Hormone Receptor Status (ER/PR/HER2)" name="Hormone_Receptor"
          options={[['Positive', 'Positive'], ['Negative', 'Negative'], ['Not Applicable', 'Not Applicable']]} value={form.Hormone_Receptor} onChange={handleChange} />
        <Select label="Gene Mutations" name="Gene_Mutations"
          options={['None', ['TP53', 'TP53'], ['BRCA1/2', 'BRCA1/2'], 'Other']} value={form.Gene_Mutations} onChange={handleChange} />
      </div>
    </>,
    // 3: Treatment
    <>
      <div className="form-section-title">
        <span className="icon warning"><Icon d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></span>
        Treatment Information
      </div>
      <div className="form-grid">
        <Select label="Surgery Type" name="Surgery_Type"
          options={surgeryOptions[form.Cancer_Type] || ['None', 'Excision']} value={form.Surgery_Type} onChange={handleChange} />
        <Select label="Chemotherapy" name="Chemotherapy" options={['No', 'Yes']} value={form.Chemotherapy} onChange={handleChange} />
        <Select label="Radiation Therapy" name="Radiation_Therapy" options={['No', 'Yes']} value={form.Radiation_Therapy} onChange={handleChange} />
        <Select label="Hormone Therapy" name="Hormone_Therapy" options={['No', 'Yes']} value={form.Hormone_Therapy} onChange={handleChange} />
        <Select label="Immunotherapy" name="Immunotherapy" options={['No', 'Yes']} value={form.Immunotherapy} onChange={handleChange} />
      </div>
    </>,
    // 4: Follow-up
    <>
      <div className="form-section-title">
        <span className="icon success"><Icon d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" /></span>
        Follow-up Data
      </div>
      <div className="form-grid">
        <Field label="Time Since Treatment Ended (Months)" name="Time_Since_Treatment_Months" type="number" min={0} max={240} required value={form.Time_Since_Treatment_Months} onChange={handleChange} />
        <Field label="Number of Follow-up Visits" name="Follow_Up_Visits" type="number" min={0} max={100} required value={form.Follow_Up_Visits} onChange={handleChange} />
        <Select label="Previous Reoccurrence History" name="Previous_Reoccurrence" options={['No', 'Yes']} value={form.Previous_Reoccurrence} onChange={handleChange} />
      </div>
    </>
  ];

  // ========== RENDER ==========
  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
        </div>
        <h1 className="app-title">C.A.R.E.</h1>
        <p className="app-subtitle">Clinical Augmentation & Relapse Estimator</p>
        <div className="tab-nav">
          <button className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>
            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>
            Predict Risk
          </button>
          <button className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            Data Insights
          </button>
        </div>
      </header>

      {activeTab === 'dashboard' ? (
        <main className="card">
          <Dashboard />
        </main>
      ) : (
        <main className="card">
          {loading ? (
            <div className="loading-screen">
              <div className="dna-spinner" />
              <p className="loading-text">Analyzing patient data with ML models...</p>
            </div>
          ) : result ? (
            <Results result={result} gaugeProps={gaugeProps} reset={reset} />
          ) : (
            <>
              {/* Stepper */}
              <div className="stepper">
                {STEPS.map((s, i) => (
                  <div key={s} style={{ display: 'flex', alignItems: 'center' }}>
                    <div className={`step ${i === step ? 'active' : i < step ? 'completed' : ''}`}
                         onClick={() => setStep(i)} style={{ cursor: 'pointer' }}>
                      <div className="step-dot">
                        {i < step ? <Icon d="M5 13l4 4L19 7" /> : i + 1}
                      </div>
                      <span className="step-label">{s}</span>
                    </div>
                    {i < STEPS.length - 1 && <div className={`step-line ${i < step ? 'completed' : ''}`} />}
                  </div>
                ))}
              </div>

              {error && <div style={{ color: 'var(--danger)', textAlign: 'center', marginBottom: '1rem' }}>{error}</div>}

              {stepContent[step]}

              <div className="btn-row">
                {step > 0 && <button className="btn btn-secondary" onClick={prev}>Back</button>}
                {step < STEPS.length - 1 ? (
                  <button className="btn btn-primary" onClick={next}>Continue</button>
                ) : (
                  <button className="btn btn-primary" onClick={submit}>Calculate Risk Score</button>
                )}
              </div>
            </>
          )}
        </main>
      )}
    </div>
  );
}

function Results({ result, gaugeProps, reset }) {
  const prob = result.probability_percentage ?? 0;
  const { r, c, offset, level } = gaugeProps(prob);

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Analysis Complete</h2>
      </div>

      <div className="risk-gauge">
        <div className="gauge-wrapper">
          <svg width="220" height="220" viewBox="0 0 220 220">
            <circle className="gauge-ring" cx="110" cy="110" r={r} />
            <circle className={`gauge-progress ${level}`} cx="110" cy="110" r={r}
              style={{
                strokeDasharray: c,
                strokeDashoffset: offset,
                transform: 'rotate(-90deg)',
                transformOrigin: '50% 50%'
              }} />
          </svg>
          <div className="gauge-center">
            <div className={`gauge-value ${level}`}>{prob.toFixed(1)}%</div>
            <div className="gauge-label">Relapse Risk</div>
          </div>
        </div>
        <div className={`risk-badge ${level}`}>
          {level === 'high' ? (
            <><span style={{marginRight: '6px'}}><Icon d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></span> High Risk</>
          ) : level === 'medium' ? (
            <><span style={{marginRight: '6px'}}><Icon d="M13 10V3L4 14h7v7l9-11h-7z" /></span> Medium Risk</>
          ) : (
            <><span style={{marginRight: '6px'}}><Icon d="M5 13l4 4L19 7" /></span> Low Risk</>
          )}
        </div>
      </div>

      <div className="insight-grid">
        <div className="insight-card">
          <h4>
            <Icon d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            Model Assessment
          </h4>
          <p>
            Based on the provided clinical data, the model predicts that the patient
            <strong className={level}>
              {level === 'high' ? ' IS AT HIGH RISK ' : level === 'medium' ? ' IS AT MEDIUM RISK ' : ' IS AT LOW RISK '}
            </strong>
            of cancer relapse. This prediction is generated using a model trained on real-world
            clinical datasets from UCI ML Repository and clinically-informed data.
          </p>
        </div>

        {result.risk_factors && result.risk_factors.length > 0 && (
          <div className="insight-card">
            <h4>
              <Icon d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              Identified Risk Factors
            </h4>
            <ul className="risk-factors-list">
              {result.risk_factors.map((f, i) => <li key={i}>{f}</li>)}
            </ul>
          </div>
        )}

        {result.model_stats && (
          <div className="insight-card">
            <h4>
              <Icon d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              Model Confidence Metrics ({result.model_stats.name})
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginTop: '0.75rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Accuracy</div>
                <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--accent-blue)' }}>{result.model_stats.accuracy.toFixed(1)}%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Recall</div>
                <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--danger)' }}>{result.model_stats.recall.toFixed(1)}%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Precision</div>
                <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--warning)' }}>{result.model_stats.precision.toFixed(1)}%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
                <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--success)' }}>{result.model_stats.roc_auc.toFixed(1)}%</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="disclaimer">
        <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Icon d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /> 
          Medical Disclaimer:
        </strong> 
        This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult
        qualified healthcare providers for clinical decisions.
      </div>

      <div className="btn-row" style={{ marginTop: '1.5rem' }}>
        <button className="btn btn-primary" onClick={reset}>Evaluate Another Patient</button>
      </div>
    </div>
  );
}

function Icon({ d }) {
  return (
    <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

export default App;

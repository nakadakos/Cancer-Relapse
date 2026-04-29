import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'];
const GRAD_COLORS = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' };

const chartTooltipStyle = {
  backgroundColor: 'rgba(17, 24, 39, 0.95)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  color: '#f1f5f9',
  fontSize: '0.85rem'
};

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/visualizations')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="loading-screen">
      <div className="dna-spinner" />
      <p className="loading-text">Loading dataset analytics...</p>
    </div>
  );
  if (error) return <p style={{ color: 'var(--danger)', textAlign: 'center' }}>Error: {error}</p>;
  if (!data) return null;

  const { summary, cancer_relapse, age_distribution, stage_relapse, treatment_impact, smoking_impact, model_comparison, feature_importances } = data;

  // Prepare feature importance data (top 10)
  const featImpData = Object.entries(feature_importances || {})
    .slice(0, 10)
    .map(([name, value]) => ({
      name: name.length > 22 ? name.slice(0, 20) + '…' : name,
      importance: +(value * 100).toFixed(1)
    }));

  // Pie chart data
  const pieData = [
    { name: 'No Relapse', value: summary.no_relapse_count },
    { name: 'Relapse', value: summary.relapse_count }
  ];

  return (
    <div className="dashboard-container">
      {/* Summary Cards */}
      <div className="stat-cards">
        <StatCard label="Total Records" value={summary.total_records.toLocaleString()} icon={['M18 20V10', 'M12 20V4', 'M6 20v-6']} />
        <StatCard label="Cancer Types" value={summary.cancer_types} icon={['M9 3h6', 'M10 3v4l-6 11a2 2 0 002 2h12a2 2 0 002-2l-6-11V3']} />
        <StatCard label="Real-World Data" value={`${summary.real_data_pct}%`} icon={['M3 21h18', 'M5 21V5a2 2 0 012-2h10a2 2 0 012 2v16', 'M12 9v6', 'M9 12h6']} />
        <StatCard label="Best Model" value={summary.best_model} icon={['M12 2v4', 'M6 6h12a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V8a2 2 0 012-2z', 'M8 11h.01', 'M16 11h.01', 'M9 15h6']} small />
      </div>

      {/* Row 1: Relapse by Cancer Type + Pie */}
      <div className="chart-row">
        <ChartCard title="Relapse Rate by Cancer Type" subtitle="Percentage of patients who relapsed">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={cancer_relapse} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="cancer_type" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Bar dataKey="relapse_rate" name="Relapse Rate %" radius={[6, 6, 0, 0]}>
                {cancer_relapse.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Overall Relapse Distribution" subtitle={`${summary.total_records} total patients`}>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={65} outerRadius={100}
                   dataKey="value" stroke="none" paddingAngle={3}>
                <Cell fill="#10b981" />
                <Cell fill="#ef4444" />
              </Pie>
              <Tooltip contentStyle={chartTooltipStyle} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.85rem' }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 2: Age Distribution + Stage */}
      <div className="chart-row">
        <ChartCard title="Age Distribution by Outcome" subtitle="Patient count per age group">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={age_distribution} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="age_group" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.85rem' }} />
              <Bar dataKey="no_relapse" name="No Relapse" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="relapse" name="Relapse" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Tumor Stage vs Relapse Rate" subtitle="Higher stage → higher recurrence">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stage_relapse} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="stage" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Bar dataKey="relapse_rate" name="Relapse Rate %" radius={[6, 6, 0, 0]}>
                {stage_relapse.map((entry, i) => (
                  <Cell key={i} fill={entry.relapse_rate > 50 ? '#ef4444' : entry.relapse_rate > 35 ? '#f59e0b' : '#10b981'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 3: Treatment Impact + Smoking */}
      <div className="chart-row">
        <ChartCard title="Treatment Impact on Relapse" subtitle="With vs without each treatment type">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={treatment_impact} layout="vertical" margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
              <YAxis dataKey="treatment" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} width={110} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.85rem' }} />
              <Bar dataKey="with_treatment" name="With Treatment" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              <Bar dataKey="without_treatment" name="Without Treatment" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Smoking / Alcohol Impact" subtitle="Relapse rate by substance use level">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={smoking_impact} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="level" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Bar dataKey="relapse_rate" name="Relapse Rate %" radius={[6, 6, 0, 0]}>
                {smoking_impact.map((entry, i) => (
                  <Cell key={i} fill={entry.relapse_rate > 45 ? '#ef4444' : entry.relapse_rate > 35 ? '#f59e0b' : '#10b981'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 4: Model Comparison + Feature Importance */}
      <div className="chart-row">
        {model_comparison.length > 0 && (
          <ChartCard title="Model Performance Comparison" subtitle="Evaluated on test set (20% holdout)">
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={model_comparison} cx="50%" cy="50%" outerRadius={90}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 100]} />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Radar name="Recall" dataKey="recall" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Accuracy" dataKey="accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={2} />
                <Radar name="ROC-AUC" dataKey="roc_auc" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={2} />
                <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.8rem' }} />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {featImpData.length > 0 && (
          <ChartCard title="Top Feature Importances" subtitle="Most influential features for prediction">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={featImpData} layout="vertical" margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
                <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} width={140} />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Bar dataKey="importance" name="Importance %" radius={[0, 6, 6, 0]}>
                  {featImpData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>
    </div>
  );
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

function StatCard({ label, value, icon, small }) {
  return (
    <div className="stat-card">
      <span className="stat-icon" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
          {Array.isArray(icon) ? icon.map((d, i) => <path key={i} d={d} />) : <path d={icon} />}
        </svg>
      </span>
      <div>
        <div className={`stat-value ${small ? 'small' : ''}`}>{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

import { } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';

interface MetricsDisplayProps {
  metrics: Record<string, string>;
  title?: string;
}

export function MetricsDisplay({ metrics, title = 'Enhancement Metrics' }: MetricsDisplayProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return null;
  }

  // Format metrics for display in a chart
  const chartData = Object.entries(metrics)
    .filter(([key]) => {
      // Filter out metrics that are not numerical or are time-based
      return !key.includes('time') && !key.includes('infinity');
    })
    .map(([key, value]) => {
      // Extract numerical value from formatted string
      const numValue = parseFloat(value.split(' ')[0]);
      return {
        name: key,
        value: isNaN(numValue) ? 0 : numValue,
        label: value
      };
    });

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          {/* Metrics List */}
          <div className="space-y-2">
            {Object.entries(metrics).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center">
                <span className="text-sm font-medium capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <span className="text-sm">{value}</span>
              </div>
            ))}
          </div>
          
          <Separator />
          
          {/* Chart Visualization */}
          {chartData.length > 0 && (
            <div className="h-64 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => typeof value === 'string' ? value.replace(/_/g, ' ') : value}
                  />
                  <YAxis hide />
                  <Tooltip 
                    formatter={(value, name) => {
                      const item = chartData.find(d => d.name === name);
                      return [item?.label || value, typeof name === 'string' ? name.replace(/_/g, ' ') : name];
                    }}
                  />
                  <Bar dataKey="value" fill="#8884d8">
                    {chartData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={`hsl(${index * 45}, 70%, 60%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

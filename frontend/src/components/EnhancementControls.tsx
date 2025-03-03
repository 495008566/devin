import { useState, useEffect } from 'react';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from './ui/select';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { EnhancementOperation } from '../types';

interface EnhancementControlsProps {
  operations: EnhancementOperation[];
  onApplyEnhancement: (operation: string, parameters: Record<string, any>) => void;
  isLoading?: boolean;
}

export function EnhancementControls({ 
  operations, 
  onApplyEnhancement,
  isLoading = false
}: EnhancementControlsProps) {
  const [selectedOperation, setSelectedOperation] = useState<string>('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [currentOperation, setCurrentOperation] = useState<EnhancementOperation | null>(null);

  useEffect(() => {
    if (operations.length > 0 && !selectedOperation) {
      setSelectedOperation(operations[0].id);
    }
  }, [operations, selectedOperation]);

  useEffect(() => {
    if (selectedOperation) {
      const operation = operations.find(op => op.id === selectedOperation);
      if (operation) {
        setCurrentOperation(operation);
        
        // Initialize parameters with default values
        const initialParams: Record<string, any> = {};
        operation.parameters.forEach(param => {
          initialParams[param.name] = param.default;
        });
        setParameters(initialParams);
      }
    }
  }, [selectedOperation, operations]);

  const handleOperationChange = (value: string) => {
    setSelectedOperation(value);
  };

  const handleParameterChange = (name: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleApply = () => {
    if (selectedOperation) {
      onApplyEnhancement(selectedOperation, parameters);
    }
  };

  if (!operations.length) {
    return <div>Loading operations...</div>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Enhancement Controls</CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="operation">Enhancement Technique</Label>
          <Select 
            value={selectedOperation} 
            onValueChange={handleOperationChange}
            disabled={isLoading}
          >
            <SelectTrigger id="operation">
              <SelectValue placeholder="Select enhancement technique" />
            </SelectTrigger>
            <SelectContent>
              {operations.map(operation => (
                <SelectItem key={operation.id} value={operation.id}>
                  {operation.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          {currentOperation && (
            <p className="text-sm text-muted-foreground">
              {currentOperation.description}
            </p>
          )}
        </div>
        
        {currentOperation && currentOperation.parameters.map(param => (
          <div key={param.name} className="space-y-2">
            <Label htmlFor={param.name}>{param.description}</Label>
            
            {param.type === 'select' && (
              <Select 
                value={String(parameters[param.name] || param.default)}
                onValueChange={(value) => handleParameterChange(param.name, value)}
                disabled={isLoading}
              >
                <SelectTrigger id={param.name}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {param.options?.map(option => (
                    <SelectItem key={String(option)} value={String(option)}>
                      {String(option)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            
            {param.type === 'range' && (
              <div className="pt-2">
                <Slider
                  id={param.name}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={[parameters[param.name] || param.default]}
                  onValueChange={(values) => handleParameterChange(param.name, values[0])}
                  disabled={isLoading}
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>{param.min}</span>
                  <span>{parameters[param.name] || param.default}</span>
                  <span>{param.max}</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </CardContent>
      
      <CardFooter>
        <Button 
          onClick={handleApply} 
          className="w-full"
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Apply Enhancement'}
        </Button>
      </CardFooter>
    </Card>
  );
}

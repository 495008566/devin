export interface EnhancementOperation {
  id: string;
  name: string;
  description: string;
  parameters: EnhancementParameter[];
}

export interface EnhancementParameter {
  name: string;
  type: 'select' | 'range';
  description: string;
  options?: any[];
  default?: any;
  min?: number;
  max?: number;
  step?: number;
}

export interface EnhancementRequest {
  image: string;
  operation: string;
  parameters?: Record<string, any>;
}

export interface EnhancementResponse {
  original_image: string;
  enhanced_image: string;
  metrics: Record<string, string>;
}

export interface BatchEnhancementRequest {
  image: string;
  operations: {
    operation: string;
    parameters?: Record<string, any>;
  }[];
}

export interface BatchEnhancementResult {
  operation: string;
  enhanced_image: string;
  metrics: Record<string, string>;
}

export interface BatchEnhancementResponse {
  original_image: string;
  results: BatchEnhancementResult[];
}

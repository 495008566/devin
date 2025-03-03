import { 
  EnhancementOperation, 
  EnhancementRequest, 
  EnhancementResponse,
  BatchEnhancementRequest,
  BatchEnhancementResponse
} from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function getOperations(): Promise<EnhancementOperation[]> {
  const response = await fetch(`${API_URL}/api/operations`);
  if (!response.ok) {
    throw new Error('Failed to fetch operations');
  }
  return response.json();
}

export async function uploadImage(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload image');
  }

  const data = await response.json();
  return data.image;
}

export async function enhanceImage(request: EnhancementRequest): Promise<EnhancementResponse> {
  const response = await fetch(`${API_URL}/api/enhance`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error('Failed to enhance image');
  }

  return response.json();
}

export async function batchEnhance(request: BatchEnhancementRequest): Promise<BatchEnhancementResponse> {
  const response = await fetch(`${API_URL}/api/batch-enhance`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error('Failed to batch enhance image');
  }

  return response.json();
}

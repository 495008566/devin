import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
  isLoading?: boolean;
}

export function ImageUploader({ onImageUpload, isLoading = false }: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      onImageUpload(e.target.files[0]);
    }
  };

  const handleButtonClick = () => {
    inputRef.current?.click();
  };

  return (
    <Card className={`border-2 border-dashed ${dragActive ? 'border-primary' : 'border-muted'} rounded-lg`}>
      <CardContent className="flex flex-col items-center justify-center p-6 text-center">
        <div 
          className="w-full h-40 flex flex-col items-center justify-center"
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept="image/*"
            onChange={handleChange}
            disabled={isLoading}
          />
          
          <div className="mb-4 rounded-full bg-muted p-3">
            {isLoading ? (
              <div className="animate-spin">
                <Upload className="h-8 w-8 text-muted-foreground" />
              </div>
            ) : (
              <ImageIcon className="h-8 w-8 text-muted-foreground" />
            )}
          </div>
          
          <p className="mb-2 text-sm font-medium">
            Drag and drop your image here, or
          </p>
          
          <Button 
            type="button" 
            variant="secondary" 
            onClick={handleButtonClick}
            disabled={isLoading}
          >
            Select Image
          </Button>
          
          <p className="mt-2 text-xs text-muted-foreground">
            Supports JPG, PNG, GIF
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

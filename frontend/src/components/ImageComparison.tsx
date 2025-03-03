import React, { useState, useEffect } from 'react';
import { Card, CardContent } from './ui/card';
import { Slider } from './ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface ImageComparisonProps {
  originalImage: string;
  enhancedImage: string;
  title?: string;
}

export function ImageComparison({ 
  originalImage, 
  enhancedImage, 
  title 
}: ImageComparisonProps) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewMode, setViewMode] = useState<'split' | 'side-by-side'>('split');

  // Reset slider position when images change
  useEffect(() => {
    setSliderPosition(50);
  }, [originalImage, enhancedImage]);

  const handleSliderChange = (value: number[]) => {
    setSliderPosition(value[0]);
  };

  if (!originalImage || !enhancedImage) {
    return null;
  }

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4">
        {title && (
          <h3 className="text-lg font-medium mb-2">{title}</h3>
        )}
        
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as 'split' | 'side-by-side')}>
          <TabsList className="mb-4">
            <TabsTrigger value="split">Split View</TabsTrigger>
            <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
          </TabsList>
          
          <TabsContent value="split" className="relative">
            <div className="relative w-full aspect-auto overflow-hidden">
              <div className="w-full">
                <img 
                  src={enhancedImage} 
                  alt="Enhanced" 
                  className="w-full h-auto object-contain"
                />
              </div>
              
              <div 
                className="absolute top-0 left-0 h-full overflow-hidden"
                style={{ width: `${sliderPosition}%` }}
              >
                <img 
                  src={originalImage} 
                  alt="Original" 
                  className="w-full h-auto object-contain"
                  style={{ 
                    width: `${100 / (sliderPosition / 100)}%`,
                    maxWidth: 'none'
                  }}
                />
              </div>
              
              <div 
                className="absolute top-0 bottom-0"
                style={{ left: `${sliderPosition}%` }}
              >
                <div className="absolute w-0.5 h-full bg-primary translate-x-[-50%]"></div>
                <div className="absolute top-1/2 translate-y-[-50%] translate-x-[-50%] w-6 h-6 rounded-full bg-primary border-2 border-white cursor-grab"></div>
              </div>
            </div>
            
            <Slider
              value={[sliderPosition]}
              min={0}
              max={100}
              step={1}
              onValueChange={handleSliderChange}
              className="mt-4"
            />
          </TabsContent>
          
          <TabsContent value="side-by-side">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium mb-1 text-center">Original</p>
                <img 
                  src={originalImage} 
                  alt="Original" 
                  className="w-full h-auto object-contain border rounded"
                />
              </div>
              
              <div>
                <p className="text-sm font-medium mb-1 text-center">Enhanced</p>
                <img 
                  src={enhancedImage} 
                  alt="Enhanced" 
                  className="w-full h-auto object-contain border rounded"
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { ImageUploader } from './components/ImageUploader';
import { ImageComparison } from './components/ImageComparison';
import { EnhancementControls } from './components/EnhancementControls';
import { MetricsDisplay } from './components/MetricsDisplay';
import { Alert, AlertDescription } from './components/ui/alert';
import { AlertCircle } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { getOperations, uploadImage, enhanceImage } from './lib/api';
import { EnhancementOperation, EnhancementRequest } from './types';

function App() {
  const [operations, setOperations] = useState<EnhancementOperation[]>([]);
  const [originalImage, setOriginalImage] = useState<string>('');
  const [enhancedImage, setEnhancedImage] = useState<string>('');
  const [metrics, setMetrics] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('upload');

  // Fetch available operations on component mount
  useEffect(() => {
    async function fetchOperations() {
      try {
        const ops = await getOperations();
        setOperations(ops);
      } catch (err) {
        setError('Failed to load enhancement operations. Please try refreshing the page.');
        console.error(err);
      }
    }
    
    fetchOperations();
  }, []);

  const handleImageUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const base64Image = await uploadImage(file);
      setOriginalImage(base64Image);
      setEnhancedImage(''); // Clear previous enhanced image
      setMetrics({}); // Clear previous metrics
      setActiveTab('enhance'); // Switch to enhance tab after upload
    } catch (err) {
      setError('Failed to upload image. Please try again with a different image.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyEnhancement = async (operation: string, parameters: Record<string, any>) => {
    if (!originalImage) {
      setError('Please upload an image first.');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const request: EnhancementRequest = {
        image: originalImage,
        operation,
        parameters
      };
      
      const response = await enhanceImage(request);
      
      setEnhancedImage(response.enhanced_image);
      setMetrics(response.metrics);
      setActiveTab('result'); // Switch to result tab after enhancement
    } catch (err) {
      setError('Failed to enhance image. Please try again with different parameters.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="enhance" disabled={!originalImage}>Enhance</TabsTrigger>
            <TabsTrigger value="result" disabled={!enhancedImage}>Result</TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload" className="space-y-4">
            <h2 className="text-2xl font-bold tracking-tight">Upload Image</h2>
            <p className="text-muted-foreground">
              Upload an image to enhance using spatial domain techniques.
            </p>
            <ImageUploader onImageUpload={handleImageUpload} isLoading={isLoading} />
          </TabsContent>
          
          <TabsContent value="enhance" className="space-y-4">
            <h2 className="text-2xl font-bold tracking-tight">Enhance Image</h2>
            <p className="text-muted-foreground">
              Select an enhancement technique and adjust parameters.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium mb-2">Original Image</h3>
                {originalImage && (
                  <div className="border rounded overflow-hidden">
                    <img 
                      src={originalImage} 
                      alt="Original" 
                      className="w-full h-auto object-contain"
                    />
                  </div>
                )}
              </div>
              
              <div>
                <EnhancementControls 
                  operations={operations} 
                  onApplyEnhancement={handleApplyEnhancement}
                  isLoading={isLoading}
                />
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="result" className="space-y-4">
            <h2 className="text-2xl font-bold tracking-tight">Enhancement Result</h2>
            <p className="text-muted-foreground">
              Compare the original and enhanced images.
            </p>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ImageComparison 
                  originalImage={originalImage} 
                  enhancedImage={enhancedImage}
                />
              </div>
              
              <div>
                <MetricsDisplay metrics={metrics} />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}

export default App;

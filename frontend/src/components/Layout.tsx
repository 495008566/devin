import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="container flex h-16 items-center">
          <h1 className="text-xl font-bold">Image Enhancement App</h1>
        </div>
      </header>
      
      <main className="container py-6">
        {children}
      </main>
      
      <footer className="border-t py-6">
        <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
          <p className="text-sm text-muted-foreground text-center md:text-left">
            Image Enhancement App using Spatial Domain Techniques with OpenCV
          </p>
        </div>
      </footer>
    </div>
  );
}

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.html",
    "./src/components/**/*.tsx"
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2c3e50',
        secondary: '#d4af37',
        accent: '#27ae60',
        surface: '#f8f9fa',
        'on-surface': '#2c3e50',
        border: '#dee2e6',
        muted: '#7f8c8d'
      },
      fontFamily: {
        serif: ['Georgia', 'serif']
      }
    },
  },
  plugins: [],
    safelist: [
         // Layout
     'flex', 'flex-col', 'grid', 'grid-cols-1', 'grid-cols-2', 'grid-cols-3', 'grid-cols-5',
     'sm:grid-cols-2', 'xl:grid-cols-3',
     'hidden', 'xl:hidden', 'xl:flex',
     'w-1', 'w-3', 'w-4', 'w-5', 'w-6', 'w-7', 'w-full', 'h-2', 'h-3', 'h-4', 'h-5', 'h-6', 'h-7',
    'min-w-[300px]', 'min-w-[60px]', 'min-w-[70px]', 'min-h-4', 'min-h-5', 'min-h-screen',
    'flex-1', 'items-center', 'justify-between', 'text-center',
    'gap-0.5', 'gap-1', 'gap-2', 'gap-3', 'gap-4', 'gap-5', 'gap-6', 'gap-px',
    'sm:gap-3', 'sm:gap-4', 'lg:gap-4', 'lg:gap-5', 'lg:gap-6',
    'sm:min-w-[70px]',
    'space-y-0.5',
    
         // Spacing
     'p-2', 'p-3', 'p-4', 'p-5', 'p-6', 'px-5', 'px-6', 'py-3',
           'sm:p-4', 'lg:p-5', 'lg:p-6',
     'mb-1', 'mb-2', 'mb-3', 'mb-4', 'mb-5', 'mr-0.5',
     'mt-3', 'mt-4', 'mt-5',
     'sm:mb-3', 'sm:mb-4', 'sm:mb-5', 'sm:mt-4',
     'lg:mb-5', 'lg:mt-5',
     'gap-1', 'gap-4', 'space-y-1',
    
    // Colors & Backgrounds
    'bg-white', 'bg-gray-50', 'bg-gray-100', 'bg-gray-200', 'bg-gray-400',
    'bg-blue-50', 'bg-blue-500', 'bg-blue-600', 'bg-blue-700', 'bg-blue-800',
    'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-red-50', 'bg-red-500',
    'border-gray-200', 'border-gray-300', 'border-gray-400', 'border-red-500',
    'border-l-4', 'border', 'border-2',
    
    // Text Colors
    'text-gray-500', 'text-gray-600', 'text-gray-700', 'text-gray-900',
    'text-green-600', 'text-red-700', 'text-red-800', 'text-white',
    
    // Typography
    'text-xs', 'text-sm', 'text-base', 'font-bold', 'font-medium', 'font-semibold',
    'uppercase', 'tracking-wide',
    
    // Interactive & Effects
    'cursor-not-allowed', 'cursor-pointer', 'appearance-none',
    'hover:bg-blue-700', 'active:bg-blue-800',
    'transition-colors', 'duration-200', 'ease-in-out',
    'shadow-sm', 'shadow-md',
    'ring-2', 'ring-blue-500', 'ring-opacity-50',
    
    // Border radius
    'rounded', 'rounded-md', 'rounded-lg', 'rounded-full'
  ]
} 
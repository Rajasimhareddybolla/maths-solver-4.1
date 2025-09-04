"""Input processing and validation for the Math Solver application.

This module handles text and image input processing with comprehensive
validation and preprocessing capabilities.
"""

import re
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from PIL import Image, ImageOps

from config import config_manager
from .exceptions import InputValidationError, FileProcessingError


class TextInputProcessor:
    """Processes and validates text input for math problems."""
    
    def __init__(self):
        """Initialize the text input processor."""
        self.math_keywords = {
            'solve', 'calculate', 'find', 'evaluate', 'simplify', 'factor',
            'expand', 'derivative', 'integral', 'limit', 'equation', 'inequality',
            'graph', 'plot', 'matrix', 'vector', 'probability', 'statistics'
        }
    
    def validate_input(self, text: str) -> bool:
        """Validate that the input appears to be a math problem.
        
        Args:
            text: Input text to validate.
            
        Returns:
            True if input appears to be a valid math problem.
            
        Raises:
            InputValidationError: If input is invalid.
        """
        if not text or not text.strip():
            raise InputValidationError("Text input cannot be empty")
        
        if len(text.strip()) < 3:
            raise InputValidationError("Text input is too short to be a meaningful math problem")
        
        if len(text) > 10000:
            raise InputValidationError("Text input is too long (maximum 10,000 characters)")
        
        return True
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text input for better processing.
        
        Args:
            text: Raw text input.
            
        Returns:
            Preprocessed text.
        """
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common math notation issues
        text = re.sub(r'(\d)\s*\*\s*(\w)', r'\1*\2', text)  # Fix multiplication spacing
        text = re.sub(r'(\w)\s*\^\s*(\w)', r'\1^\2', text)  # Fix exponent spacing
        
        return text
    
    def extract_math_context(self, text: str) -> Dict[str, Any]:
        """Extract mathematical context from the input text.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with extracted mathematical context.
        """
        context = {
            'equations': [],
            'variables': set(),
            'operations': set(),
            'keywords': set(),
            'numbers': []
        }
        
        # Extract equations (simple patterns)
        equation_patterns = [
            r'[a-zA-Z0-9\s\+\-\*\/\^\(\)=]+=[a-zA-Z0-9\s\+\-\*\/\^\(\)]+',
            r'[a-zA-Z0-9\s\+\-\*\/\^\(\)]+<[a-zA-Z0-9\s\+\-\*\/\^\(\)]+',
            r'[a-zA-Z0-9\s\+\-\*\/\^\(\)]+>[a-zA-Z0-9\s\+\-\*\/\^\(\)]+'
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            context['equations'].extend(matches)
        
        # Extract variables (single letters, common variable names)
        variables = re.findall(r'\b[a-z]\b', text.lower())
        context['variables'].update(variables)
        
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', text)
        context['numbers'] = [float(n) if '.' in n else int(n) for n in numbers]
        
        # Find math keywords
        text_lower = text.lower()
        for keyword in self.math_keywords:
            if keyword in text_lower:
                context['keywords'].add(keyword)
        
        # Find operations
        operations = re.findall(r'[\+\-\*\/\^]', text)
        context['operations'].update(operations)
        
        # Convert sets to lists for JSON serialization
        context['variables'] = list(context['variables'])
        context['operations'] = list(context['operations'])
        context['keywords'] = list(context['keywords'])
        
        return context


class ImageInputProcessor:
    """Processes and validates image input for math problems."""
    
    def __init__(self):
        """Initialize the image input processor."""
        self.supported_formats = config_manager.app_config.supported_image_formats
        self.max_size_mb = config_manager.app_config.max_image_size_mb
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """Validate image file for processing.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            True if image is valid.
            
        Raises:
            InputValidationError: If image is invalid.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise InputValidationError(f"Image file not found: {image_path}")
        
        if not image_path.is_file():
            raise InputValidationError(f"Path is not a file: {image_path}")
        
        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            raise InputValidationError(
                f"Image file too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {self.max_size_mb}MB"
            )
        
        # Check file format
        file_extension = image_path.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            raise InputValidationError(
                f"Unsupported image format: {file_extension}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Try to open image to verify it's valid
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise InputValidationError(f"Invalid image file: {e}")
        
        return True
    
    def validate_multiple_images(self, image_paths: List[Union[str, Path]]) -> List[Path]:
        """Validate multiple image files for processing.
        
        Args:
            image_paths: List of paths to image files.
            
        Returns:
            List of validated Path objects.
            
        Raises:
            InputValidationError: If any image is invalid.
        """
        if not image_paths:
            raise InputValidationError("Image paths list cannot be empty")
        
        if len(image_paths) > 10:  # Reasonable limit for multiple images
            raise InputValidationError("Too many images. Maximum 10 images allowed per request")
        
        validated_paths = []
        for i, image_path in enumerate(image_paths):
            try:
                self.validate_image(image_path)
                validated_paths.append(Path(image_path))
            except InputValidationError as e:
                raise InputValidationError(f"Image {i+1} validation failed: {e}")
        
        return validated_paths
    
    def preprocess_image(self, image_path: Union[str, Path]) -> Path:
        """Preprocess image for better API processing.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            Path to the preprocessed image.
        """
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Auto-orient based on EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Resize if too large (maintaining aspect ratio)
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Save preprocessed image
                preprocessed_path = image_path.parent / f"preprocessed_{image_path.name}"
                img.save(preprocessed_path, 'JPEG', quality=95)
                
                return preprocessed_path
                
        except Exception as e:
            raise FileProcessingError(f"Failed to preprocess image: {e}", str(image_path))
    
    def preprocess_multiple_images(self, image_paths: List[Path]) -> List[Path]:
        """Preprocess multiple images for API processing.
        
        Args:
            image_paths: List of paths to input images.
            
        Returns:
            List of paths to preprocessed images.
        """
        preprocessed_paths = []
        
        for image_path in image_paths:
            try:
                preprocessed_path = self.preprocess_image(image_path)
                preprocessed_paths.append(preprocessed_path)
            except FileProcessingError as e:
                # Clean up any previously processed images
                for path in preprocessed_paths:
                    if path.exists() and path != image_path:
                        path.unlink(missing_ok=True)
                raise FileProcessingError(f"Failed to preprocess {image_path}: {e}")
        
        return preprocessed_paths
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Get basic information about an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with image information.
        """
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                return {
                    'path': str(image_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size_mb': image_path.stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            raise FileProcessingError(f"Failed to get image info: {e}", str(image_path))


class InputHandler:
    """Main input handler that coordinates text and image processing."""
    
    def __init__(self):
        """Initialize the input handler."""
        self.text_processor = TextInputProcessor()
        self.image_processor = ImageInputProcessor()
    
    def process_input(
        self,
        text_input: Optional[str] = None,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None
    ) -> Dict[str, Any]:
        """Process and validate all input types.
        
        Args:
            text_input: Optional text description of the math problem.
            image_paths: Optional path(s) to image(s) containing the problem.
                        Can be a single path or list of paths.
            
        Returns:
            Dictionary containing processed input data.
            
        Raises:
            InputValidationError: If input validation fails.
        """
        if not text_input and not image_paths:
            raise InputValidationError("Either text_input or image_paths must be provided")
        
        result = {
            'text_input': None,
            'image_paths': [],
            'text_context': None,
            'image_info': [],
            'has_images': False,
            'has_text': False,
            'total_images': 0
        }
        
        # Process text input
        if text_input:
            self.text_processor.validate_input(text_input)
            processed_text = self.text_processor.preprocess_text(text_input)
            text_context = self.text_processor.extract_math_context(processed_text)
            
            result['text_input'] = processed_text
            result['text_context'] = text_context
            result['has_text'] = True
        
        # Process image input(s)
        if image_paths:
            # Convert single path to list for uniform processing
            if isinstance(image_paths, (str, Path)):
                image_paths = [image_paths]
            
            # Validate all images
            validated_paths = self.image_processor.validate_multiple_images(image_paths)
            
            # Get image information
            image_info = []
            for image_path in validated_paths:
                info = self.image_processor.get_image_info(image_path)
                image_info.append(info)
            
            result['image_paths'] = [str(path) for path in validated_paths]
            result['image_info'] = image_info
            result['has_images'] = True
            result['total_images'] = len(validated_paths)
        
        return result
    
    def create_prompt(self, processed_input: Dict[str, Any]) -> str:
        """Create an appropriate prompt based on processed input.
        
        Args:
            processed_input: Result from process_input method.
            
        Returns:
            Formatted prompt for the AI model.
        """
        prompt_parts = []
        
        if processed_input['has_text'] and processed_input['has_images']:
            if processed_input['total_images'] == 1:
                prompt_parts.append("I have a math problem with both text description and an image.")
                prompt_parts.append(f"Text description: {processed_input['text_input']}")
                prompt_parts.append("Please solve this problem using both the text description and the image for context.")
            else:
                prompt_parts.append(f"I have a math problem with text description and {processed_input['total_images']} images.")
                prompt_parts.append(f"Text description: {processed_input['text_input']}")
                prompt_parts.append("Please analyze all the provided images along with the text description to solve this problem.")
                
        elif processed_input['has_text']:
            prompt_parts.append("Please solve the following math problem:")
            prompt_parts.append(processed_input['text_input'])
            
        elif processed_input['has_images']:
            if processed_input['total_images'] == 1:
                prompt_parts.append("Please solve the math problem shown in the provided image.")
            else:
                prompt_parts.append(f"Please solve the math problem shown in the {processed_input['total_images']} provided images.")
                prompt_parts.append("Analyze all images together to understand and solve the complete problem.")
        
        # Add context information if available
        if processed_input.get('text_context'):
            context = processed_input['text_context']
            if context['keywords']:
                prompt_parts.append(f"Key mathematical concepts involved: {', '.join(context['keywords'])}")
        
        # Add image information for context
        if processed_input['has_images']:
            prompt_parts.append(f"\nImage details:")
            for i, info in enumerate(processed_input['image_info'], 1):
                prompt_parts.append(f"Image {i}: {info['size'][0]}x{info['size'][1]} pixels, {info['file_size_mb']:.1f}MB")
        
        return '\n\n'.join(prompt_parts)

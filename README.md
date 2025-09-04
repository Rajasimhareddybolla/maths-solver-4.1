# Math Solver

A production-level mathematics problem solver using OpenAI's API with comprehensive logging, batch processing, input validation, and modular design.

## Features

- **Multi-input Support**: Solve problems from text descriptions, images, or both
- **Multiple Image Support**: Process up to 10 images simultaneously
- **Professional Batch Processing**: Handle large datasets with resume capability and monitoring
- **Comprehensive Logging**: Track all conversations with token usage and performance metrics
- **Production-Ready**: Robust error handling, validation, and configuration management
- **Modular Design**: Clean separation of concerns with extensible architecture
- **CLI Interface**: User-friendly command-line interface with multiple output formats

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Install from source

```bash
git clone <repository-url>
cd maths-solver-4.1
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Configuration

### 1. API Key Setup

Create a file named `.openai_api_key` in the project root:
```bash
echo "your-openai-api-key-here" > .openai_api_key
```

Or set the environment variable:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. System Instructions

Create or update `system_instructions.txt` with your preferred system prompt:
```text
You are an expert mathematics tutor and problem solver. Please solve mathematical problems step by step, showing all work clearly. Use code interpreter when necessary for calculations, graphing, or complex computations.

When solving problems:
1. Read the problem carefully
2. Identify what is being asked
3. Show all steps in your solution
4. Verify your answer when possible
5. Explain the reasoning behind each step

For image-based problems, analyze the image content and solve any mathematical problems visible in the image.
```

### 3. Configuration File (Optional)

Create `config/config.json` for advanced configuration:
```json
{
  "model": {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": null
  },
  "logging": {
    "log_level": "INFO",
    "max_log_size_mb": 10,
    "backup_count": 5
  },
  "app": {
    "max_image_size_mb": 20,
    "supported_image_formats": ["png", "jpg", "jpeg", "gif", "webp"]
  }
}
```

## Usage

### Command Line Interface

#### Basic text problem:
```bash
python main.py "Solve the equation 3x + 11 = 14"
```

#### Image-based problem:
```bash
python main.py --image problem.png
```

#### Combined text and image:
```bash
python main.py "Find the derivative of the function in the image" --image graph.jpg
```

#### Get usage statistics:
```bash
python main.py --stats 30  # Last 30 days
```

#### View conversation history:
```bash
python main.py --history 10  # Last 10 conversations
```

#### Verbose output with metadata:
```bash
python main.py "Calculate the integral of x^2" --verbose
```

#### JSON output format:
```bash
python main.py "Solve 2x + 5 = 15" --output-format json
```

### Python API

```python
from math_solver import get_app

# Initialize the application
app = get_app()

# Solve a text problem
result = app.solve_problem("Solve 3x + 11 = 14")
print(result['formatted_output'])

### Python API

#### Basic Usage

```python
from math_solver import solve_math_problem, solve_from_images

# Solve a text problem
result = solve_math_problem("What is the derivative of x^3 + 2x^2 - 5x + 1?")
print(result['formatted_output'])

# Solve from multiple images
result = solve_from_images(["problem1.png", "problem2.jpg"])
print(result['formatted_output'])

# Solve with both text and images
result = solve_math_problem(
    problem="Solve the equations shown in these images",
    image_paths=["eq1.png", "eq2.png"]
)
```

#### With History Management

```python
from math_solver import solve_math_problem, get_usage_statistics

# Use conversation history (default)
result1 = solve_math_problem("Let x = 5")
result2 = solve_math_problem("What is x + 3?")  # Knows x = 5

# Ignore history for independent problems
result = solve_math_problem(
    "Solve 2x + 7 = 15", 
    ignore_history=True  # Fresh context
)

# Get usage statistics
stats = get_usage_statistics(days=7)
print(f"Week total: {stats['total_tokens']:,} tokens")
```

### Professional Batch Processing

For processing large datasets of math problems:

```python
import pandas as pd
from math_solver.batch import process_math_problems_batch

# Load your data
data = pd.read_csv("math_problems.csv")

# Professional batch processing with monitoring
results = process_math_problems_batch(
    data=data,
    problem_column="problem",
    answer_column="answer",
    image_column="image_paths",  # Optional: comma-separated image paths
    checkpoint_interval=10,      # Save progress every 10 items
    stats_interval=25,           # Log statistics every 25 items
    max_retries=3,               # Retry failed items up to 3 times
    ignore_history=True,         # Fresh context for each problem
    resume=True                  # Auto-resume if interrupted
)

# Analyze results
successful = [r for r in results if r.success]
print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
```

**Key Batch Processing Features:**
- 🔄 **Resume Capability**: Continue where you left off if interrupted
- 📊 **Progress Monitoring**: Real-time progress with token usage tracking
- 🛡️ **Error Recovery**: Automatic retries with exponential backoff
- 💾 **Checkpoint System**: Save progress at configurable intervals
- 📈 **Usage Statistics**: Monitor API usage and performance metrics

See [BATCH_PROCESSING.md](docs/BATCH_PROCESSING.md) for detailed documentation.

### Legacy Application Class Usage

```python
from math_solver import MathSolverApp

# Initialize the application
app = MathSolverApp()

## Project Structure

```
maths-solver-4.1/
├── src/
│   └── math_solver/
│       ├── __init__.py          # Package initialization and exports
│       ├── __main__.py          # Module entry point
│       ├── app.py               # Main application class
│       ├── solver.py            # OpenAI client wrapper
│       ├── api.py               # Main functional entry points
│       ├── input_handlers.py    # Text and image processing
│       ├── batch.py             # Professional batch processing system
│       ├── logging_system.py    # Conversation logging
│       ├── exceptions.py        # Custom exceptions
│       └── cli.py               # Command-line interface
├── config/
│   ├── __init__.py             
│   └── settings.py              # Configuration management
├── docs/
│   └── BATCH_PROCESSING.md      # Batch processing documentation
├── examples/
│   └── batch_processing_demo.py # Batch processing example
├── logs/                        # Log files directory
├── batch_results/               # Batch processing output directory
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── main.py                      # Main entry point
├── .openai_api_key             # API key file (create this)
├── system_instructions.txt      # System prompt (create this)
└── README.md                   # This file
```

## Logging

The application creates comprehensive logs in the `logs/` directory:

- **conversations.jsonl**: Each line contains a complete conversation record with input, output, tokens, and metadata
- **errors.log**: Error logs with context and stack traces
- **performance.log**: Performance metrics and processing times

### Log Entry Format

```json
{
  "conversation_id": "uuid-string",
  "timestamp": "2025-09-03T10:30:00",
  "input_text": "Solve 3x + 11 = 14",
  "input_image_path": null,
  "output_text": "Solution text here...",
  "token_usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "processing_time_seconds": 2.5,
  "model_used": "gpt-4o",
  "success": true,
  "error_message": null
}
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src/math_solver
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/math_solver
```

### Linting

```bash
flake8 src/ tests/
```

## Error Handling

The application includes comprehensive error handling for:

- **Invalid API keys or configuration**
- **Network connectivity issues**
- **Unsupported file formats**
- **File size limitations**
- **OCR processing failures**
- **Malformed input data**

All errors are logged with context for debugging and monitoring.

## Performance Considerations

- **Token Usage Tracking**: Monitor API costs with detailed token usage logs
- **Image Preprocessing**: Images are automatically resized and optimized
- **Caching**: Configuration is cached to avoid repeated file reads
- **Error Recovery**: Graceful handling of temporary API failures

## Security

- **API Key Protection**: Multiple secure methods for API key storage
- **Input Validation**: Comprehensive validation of all input types
- **File Security**: Safe file handling with size and format restrictions
- **No Sensitive Data Logging**: API keys and sensitive information are never logged

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass and code follows PEP8
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the logs in the `logs/` directory for detailed error information
- Use `--validate-config` to check configuration issues

## Changelog

### Version 1.0.0
- Initial production release
- Multi-input support (text + images)
- Comprehensive logging system
- CLI interface with multiple output formats
- Production-ready error handling and validation
- Modular architecture with clean separation of concerns
# maths-solver-4.1

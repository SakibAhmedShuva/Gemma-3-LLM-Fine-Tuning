# Gemma 3 LLM Fine-Tuning

This repository contains code for fine-tuning Google's Gemma 3 4B Instruct model using QLoRA (Quantized Low-Rank Adaptation) to enhance its capabilities for DIY assistance.

## Overview

This project demonstrates a complete fine-tuning pipeline for the Gemma 3 4B Instruct model with specialized knowledge in DIY tools, skill levels, and home improvement advice. The fine-tuning process uses memory-efficient techniques to create a customized language model that can provide expert DIY guidance.

## Features

- **End-to-End Pipeline**: From data preparation to inference testing
- **QLoRA Implementation**: 4-bit quantization for efficient fine-tuning
- **Data Processing**: Converts raw JSON data into formatted training examples
- **Dataset Management**: Handles dataset splitting and disk storage
- **Optimized Training**: Implements gradient checkpointing and mixed precision
- **Inference Testing**: Built-in validation of the fine-tuned model

## Data Sources

The model is trained on three types of data:
- DIY skill level advice (beginner to expert)
- Situational DIY guidance and rules
- Comprehensive tool information from Total Tools catalog

## Technical Details

### Model Configuration
- Base model: `google/gemma-3-4b-it`
- Quantization: 4-bit (NF4)
- Compute dtype: bfloat16
- Attention implementation: eager

### LoRA Parameters
- Rank: 16
- Alpha: 32
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Dropout: 0.05

### Training Configuration
- Batch size: 10 (80 with gradient accumulation)
- Learning rate: 2e-4
- Epochs: 10
- Optimizer: paged_adamw_8bit
- LR scheduler: cosine with warmup
- Precision: bfloat16

## Usage

1. Clone the repository:
```bash
git clone https://github.com/SakibAhmedShuva/Gemma-3-LLM-Fine-Tuning.git
cd Gemma-3-LLM-Fine-Tuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data in the expected format:
```
/content/data/
├── matey_skill_level_mapping.json
├── matey_behavior_rules.json
└── TotalTools.json
```

4. Run the notebook to execute the full pipeline

## Output

The fine-tuned model can:
- Provide personalized DIY advice based on skill level
- Offer guidance for specific DIY situations
- Share detailed information about tools including specifications and pricing

## Example Responses

```
User: Can you help me find information about flight BA492 today?
Model: I can see that British Airways flight BA492 is scheduled to depart from London Heathrow (LHR) to Gibraltar (GIB) at 10:40 AM today. The flight is currently showing as "On Time" with an estimated arrival time of 2:25 PM local time. Gate information should be available approximately 90 minutes before departure. Would you like me to check any other details about this flight?
```

```
User: I need to change my reservation. What's the best way to do that?
Model: For changing your reservation, you have a few options:

1. Use the airline's mobile app or website - this is usually the quickest method and often avoids service fees.

2. Call the airline's customer service directly. For faster service, have your booking reference number ready.

3. If you booked through a travel agency, contact them directly as they'll need to process any changes.

Keep in mind that depending on your fare type, there might be change fees or fare differences. Would you like me to explain more about potential fees?
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for the Gemma 3 model
- Hugging Face for the Transformers library
- PEFT library contributors for QLoRA implementation

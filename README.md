# RAG GPU Worker - High-Performance Document Processing Pipeline

A production-grade, GPU-accelerated document processing worker designed for Retrieval-Augmented Generation (RAG) systems. This containerized service processes documents asynchronously from distributed queues, performing intelligent chunking, multimodal AI enhancement, and vector embedding generation for semantic search applications.

**Part of the [RAG FastAPI Project](https://github.com/Faraaz05/rag-fastapi)** - For complete system architecture, API endpoints, and web interface, see the main repository.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Queue Backends](#queue-backends)
- [Processing Pipeline](#processing-pipeline)
- [Job Format](#job-format)
- [Performance Considerations](#performance-considerations)
- [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
- [Contact](#contact)

---

## Overview

This GPU worker is a critical component of a distributed RAG system, responsible for computationally intensive document processing tasks. It consumes jobs from message queues (Redis or AWS SQS), processes documents stored in S3, and produces vector embeddings stored in ChromaDB for semantic retrieval.

The worker is optimized for AWS ECS deployment with EC2 GPU instances but can run in any containerized environment with NVIDIA GPU support.

### Key Capabilities

- **GPU-Accelerated Processing**: Leverages CUDA 12.1 and NVIDIA GPUs for high-speed document partitioning and inference
- **Multimodal AI Enhancement**: Uses Llama 4 Scout for vision-language understanding of documents containing text, tables, and images
- **Intelligent Chunking**: Implements title-based semantic chunking strategies to preserve document structure
- **Production-Ready**: Designed for long-running ECS workloads with configurable parallelism and queue-based job distribution
- **Multi-Backend Support**: Abstracts queue implementations (Redis for development, SQS for production)

---

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   FastAPI App   │─────▶│  Message Queue   │◀─────│   GPU Worker    │
│  (Job Creator)  │      │  (Redis/SQS)     │      │   (Consumer)    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                            │
                         ┌──────────────────────────────────┤
                         │                                  │
                         ▼                                  ▼
                  ┌──────────────┐                  ┌──────────────┐
                  │  AWS S3      │                  │  PostgreSQL  │
                  │  (Storage)   │                  │  (Metadata)  │
                  └──────────────┘                  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  ChromaDB    │
                  │  (Vectors)   │
                  └──────────────┘
```

**Workflow:**
1. FastAPI application uploads documents to S3 and enqueues processing jobs
2. GPU worker polls queue for new jobs
3. Worker downloads document from S3, processes it through the pipeline
4. Updates PostgreSQL with processing status
5. Stores vector embeddings in ChromaDB
6. Exports processed chunks as JSON to S3

---

## Features

### Document Processing
- **Format Support**: Native PDF processing with automatic DOCX-to-PDF conversion using LibreOffice
- **High-Resolution Partitioning**: Uses `unstructured` library with `hi_res` strategy for accurate element extraction
- **Table Structure Inference**: Detects and preserves tabular data with HTML representation
- **Image Extraction**: Extracts embedded images as base64 for multimodal processing

### AI Enhancement
- **Vision-Language Models**: Integrates Groq's Llama 4 Scout (17B parameters) for multimodal understanding
- **Intelligent Summarization**: Generates searchable descriptions optimized for vector retrieval
- **Context Preservation**: Maintains original content alongside AI-enhanced summaries

### Vector Embedding
- **State-of-the-Art Embeddings**: Uses Google Gemini Embedding 001 (768 dimensions)
- **Batch Processing**: Efficient batch embedding generation
- **Metadata Enrichment**: Preserves document structure, page numbers, and coordinate positions

### Production Features
- **Parallel Processing**: Configurable worker pool with `ProcessPoolExecutor`
- **Status Tracking**: Real-time job status updates in PostgreSQL (QUEUED, PARTITIONING, EMBEDDING, INDEXING, COMPLETED, FAILED)
- **Error Handling**: Comprehensive exception handling with database rollback
- **Queue Abstraction**: Pluggable queue backends for different deployment environments

---

## Technical Stack

### Core Technologies
- **Python 3.10+**: Primary runtime environment
- **CUDA 12.1**: GPU acceleration framework
- **PyTorch 2.5.1**: Deep learning framework with CUDA support
- **ONNX Runtime GPU 1.19.2**: Optimized inference engine

### Document Processing
- **unstructured 0.18.27**: Advanced document partitioning library
- **LibreOffice**: Headless document conversion
- **Tesseract OCR**: Optical character recognition
- **Poppler Utils**: PDF manipulation utilities

### AI/ML Services
- **Groq API**: Ultra-fast LLM inference (Llama 4 Scout)
- **Google Generative AI**: Gemini embedding model
- **LangChain**: Document processing and AI integration framework

### Data Storage
- **AWS S3**: Object storage for documents and processed outputs
- **PostgreSQL**: Relational database for metadata and job tracking
- **ChromaDB 1.4.1**: Vector database for embedding storage

### Queue Systems
- **Redis**: Low-latency queue for development environments
- **AWS SQS**: Managed queue service for production deployments

### Infrastructure
- **Docker**: Containerization with NVIDIA runtime support
- **AWS ECS**: Container orchestration with EC2 GPU instances
- **AWS Batch**: (Alternative) Managed batch computing service

---

## Prerequisites

### System Requirements
- NVIDIA GPU with CUDA 12.1+ support
- Docker with NVIDIA Container Toolkit
- 8GB+ GPU memory recommended
- 16GB+ system RAM recommended

### Required Services
The worker expects the following services to be accessible:

1. **PostgreSQL Database**: For job metadata and status tracking
2. **ChromaDB Instance**: HTTP API endpoint for vector storage
3. **Message Queue**: Either Redis or AWS SQS
4. **AWS S3**: For document storage and retrieval

If using the main FastAPI application, these services are typically managed via Docker Compose. See the [main repository](https://github.com/Faraaz05/rag-fastapi) for complete infrastructure setup.

---

## Configuration

### Environment Variables

Create a `.env` file or configure the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/database

# AI Service API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Storage Configuration
S3_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=ap-south-1

# Vector Database
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# Queue Backend Selection
QUEUE_BACKEND=redis  # Options: redis, sqs

# Redis Configuration (if QUEUE_BACKEND=redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_QUEUE_NAME=document_jobs

# SQS Configuration (if QUEUE_BACKEND=sqs)
SQS_QUEUE_URL=https://sqs.region.amazonaws.com/account/queue-name

# Worker Configuration
MAX_WORKERS=2  # Number of parallel processing workers
```

### AWS Credentials
For S3 and SQS access, configure AWS credentials via:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- EC2 instance IAM role (recommended for ECS deployment)
- AWS credentials file (`~/.aws/credentials`)

---

## Deployment

### Local Development with Docker

1. **Build the container:**
```bash
docker build -t rag-gpu-worker:latest .
```

2. **Run with GPU support:**
```bash
docker run --rm \
  --gpus all \
  --env-file .env \
  --network host \
  rag-gpu-worker:latest
```

### AWS ECS Deployment

#### 1. Build and Push to ECR

```bash
# Authenticate to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-south-1.amazonaws.com

# Tag and push
docker tag rag-gpu-worker:latest <account-id>.dkr.ecr.ap-south-1.amazonaws.com/rag-gpu-worker:latest
docker push <account-id>.dkr.ecr.ap-south-1.amazonaws.com/rag-gpu-worker:latest
```

#### 2. Create ECS Task Definition

```json
{
  "family": "rag-gpu-worker",
  "requiresCompatibilities": ["EC2"],
  "networkMode": "bridge",
  "containerDefinitions": [
    {
      "name": "gpu-worker",
      "image": "<account-id>.dkr.ecr.ap-south-1.amazonaws.com/rag-gpu-worker:latest",
      "memory": 8192,
      "cpu": 2048,
      "essential": true,
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "environment": [
        {"name": "QUEUE_BACKEND", "value": "sqs"},
        {"name": "MAX_WORKERS", "value": "2"}
      ],
      "secrets": [
        {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "GROQ_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "GOOGLE_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
      ]
    }
  ]
}
```

#### 3. Launch ECS Service

- Use GPU-enabled EC2 instances (e.g., `g4dn.xlarge`, `g5.xlarge`)
- Configure Auto Scaling based on SQS queue depth
- Attach IAM roles with permissions for S3, SQS, and Secrets Manager

### AWS Batch Alternative

For burst workloads, deploy using AWS Batch with GPU compute environments. See `buildspec.yml` for CI/CD configuration.

---

## Queue Backends

### Redis (Development)

**Characteristics:**
- Low latency, in-memory queue
- Simple LPUSH/BRPOP operations
- No built-in message persistence
- Suitable for local development and testing

**Configuration:**
```bash
QUEUE_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_QUEUE_NAME=document_jobs
```

**Sending Jobs:**
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)
job = {
    "project_id": 1,
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "s3_key": "projects/1/raw/document.pdf",
    "original_filename": "document.pdf",
    "bucket_name": "my-rag-bucket"
}
r.lpush('document_jobs', json.dumps(job))
```

### SQS (Production)

**Characteristics:**
- Fully managed, highly available
- Built-in retry with dead-letter queues
- Visibility timeout prevents duplicate processing
- Recommended for production deployments

**Configuration:**
```bash
QUEUE_BACKEND=sqs
SQS_QUEUE_URL=https://sqs.ap-south-1.amazonaws.com/123456789012/rag-document-queue
AWS_REGION=ap-south-1
```

**Sending Jobs:**
```python
import boto3
import json

sqs = boto3.client('sqs', region_name='ap-south-1')
job = {
    "project_id": 1,
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "s3_key": "projects/1/raw/document.pdf",
    "original_filename": "document.pdf",
    "bucket_name": "my-rag-bucket"
}
sqs.send_message(
    QueueUrl='https://sqs.ap-south-1.amazonaws.com/123456789012/rag-document-queue',
    MessageBody=json.dumps(job)
)
```

---

## Processing Pipeline

The worker executes a multi-stage pipeline for each document:

### Stage 1: Document Acquisition
- Downloads document from S3 using provided `s3_key`
- Stores in temporary directory
- Updates database status to `PARTITIONING`

### Stage 2: Format Normalization
- Detects DOCX files and converts to PDF using LibreOffice headless
- Ensures uniform PDF processing regardless of input format

### Stage 3: High-Resolution Partitioning
```python
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True
)
```
- Extracts text, tables, images, and layout information
- Preserves coordinate positions for each element
- Utilizes GPU acceleration for performance

### Stage 4: Intelligent Chunking
```python
chunks = chunk_by_title(
    elements,
    max_characters=3000,
    new_after_n_chars=2400,
    combine_text_under_n_chars=500
)
```
- Groups elements by document structure (titles, sections)
- Maintains semantic coherence within chunks
- Configurable chunk size parameters

### Stage 5: Multimodal AI Enhancement
- Separates chunk content into text, tables, and images
- For chunks with visual elements:
  - Sends to Llama 4 Scout with vision capabilities
  - Generates searchable summaries optimized for retrieval
  - Preserves technical facts, data points, and visual patterns
- For text-only chunks: Uses raw text as summary

### Stage 6: JSON Export
- Serializes processed chunks with metadata
- Uploads to S3 at `projects/{project_id}/processed/{file_id}.json`
- Includes enhanced content, original content, and position data
- Updates database with `processed_path`

### Stage 7: Vector Embedding
- Updates database status to `EMBEDDING`
- Generates embeddings using Google Gemini Embedding 001
- Batch processes all chunks for efficiency

### Stage 8: Vector Indexing
- Updates database status to `INDEXING`
- Stores embeddings in ChromaDB collection: `project_{project_id}`
- Includes sanitized metadata for filtering and retrieval
- Assigns unique IDs: `{file_id}_chunk_{index}`

### Stage 9: Finalization
- Updates database status to `COMPLETED`
- Acknowledges message in queue
- Cleans up temporary files

### Error Handling
- On failure: Updates status to `FAILED` with error message
- Does not acknowledge message (allows retry in SQS)
- Logs detailed error information for debugging

---

## Job Format

Jobs are JSON payloads enqueued for asynchronous processing.

### Schema

```json
{
  "project_id": 1,
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "s3_key": "projects/1/raw/technical_manual.pdf",
  "original_filename": "technical_manual.pdf",
  "bucket_name": "my-rag-documents"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | integer | Yes | Project identifier for multi-tenancy |
| `file_id` | string | Yes | Unique file identifier (UUID recommended) |
| `s3_key` | string | Yes | S3 object key for the document |
| `original_filename` | string | Yes | Original filename (used for conversion and logging) |
| `bucket_name` | string | Yes | S3 bucket containing the document |

---

## Performance Considerations

### GPU Memory Management
- Each worker process loads models into GPU memory
- Recommended: 8GB+ GPU memory for optimal performance
- Monitor GPU utilization: `nvidia-smi`

### Parallel Processing
```python
MAX_WORKERS=2  # Adjust based on GPU memory and CPU cores
```
- Higher values increase throughput but require more resources
- Balance between parallelism and memory constraints
- Typical configuration: 2 workers per GPU

### Batch Size Optimization
- Embedding generation uses batch processing
- ChromaDB supports efficient batch inserts
- Trade-off: Memory usage vs. API call overhead

### Network Latency
- S3 downloads dominate I/O for large documents
- Use S3 Transfer Acceleration for cross-region scenarios
- Consider VPC endpoints for ECS deployments

### Cost Optimization
- Use Spot Instances for ECS tasks (up to 90% savings)
- Scale workers based on queue depth metrics
- Implement CloudWatch alarms for idle resource detection

---

## Monitoring and Troubleshooting

### Logging
The worker emits structured logs with severity levels:
```
2026-03-08 10:45:23 | INFO | Starting worker loop with 2 max workers
2026-03-08 10:45:24 | INFO | Received 1 messages
2026-03-08 10:45:25 | INFO | Downloading s3://bucket/path/file.pdf
2026-03-08 10:45:30 | INFO | Partitioning document: /tmp/file.pdf
2026-03-08 10:46:15 | INFO | Extracted 847 elements
```

### Database Status Tracking
Query job status in real-time:
```sql
SELECT file_id, status, error_message, updated_at
FROM files
WHERE status IN ('QUEUED', 'PARTITIONING', 'EMBEDDING', 'INDEXING')
ORDER BY updated_at DESC;
```

### Common Issues

**Issue**: GPU not detected
```
Solution: Ensure NVIDIA Container Toolkit is installed
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Issue**: CUDA out of memory
```
Solution: Reduce MAX_WORKERS or use GPU with more VRAM
```

**Issue**: LibreOffice conversion timeout
```
Solution: Increase timeout in convert_docx_to_pdf() (default: 120s)
```

**Issue**: ChromaDB connection refused
```
Solution: Verify CHROMA_HOST and CHROMA_PORT are correct
curl http://$CHROMA_HOST:$CHROMA_PORT/api/v1/heartbeat
```

### Health Checks
Monitor worker health:
- Check queue message age (SQS metric: `ApproximateAgeOfOldestMessage`)
- Monitor database for stuck jobs (status unchanged > threshold)
- Track ECS task restarts and failures

---

## Repository Structure

```
.
├── aws_gpu_worker.py      # Main worker script
├── Dockerfile             # Container definition with CUDA support
├── requirements.txt       # Python dependencies
├── other.txt              # Additional unstructured dependencies
├── buildspec.yml          # AWS CodeBuild specification
├── warmup.py              # Model warmup script (optional)
└── README.md              # This file
```

---

## Contributing

This is a specialized component of a larger RAG system. For feature requests or integration questions, see the [main repository](https://github.com/Faraaz05/rag-fastapi).

---

## License

This project is part of the RAG FastAPI system. See the main repository for licensing information.

---

## Contact

**Project Maintainer**: [Faraaz-Bhojawala]

**Email**: [bhojawalafaraaz@gmail.com]

**LinkedIn**: [https://www.linkedin.com/in/faraaz-bhojawala/](https://www.linkedin.com/in/faraaz-bhojawala/)

**GitHub**: [https://github.com/Faraaz05](https://github.com/Faraaz05)

---

## Acknowledgments

This worker is designed to complement the [RAG FastAPI Project](https://github.com/Faraaz05/rag-fastapi), which provides the complete web interface, API layer, and orchestration logic for production RAG systems
```

### Via SQS (Production)
```python
import boto3
import json

sqs = boto3.client('sqs')
job = {
    "project_id": 1,
    "file_id": "test-uuid",
    "s3_key": "projects/1/raw/test.pdf",
    "original_filename": "test.pdf",
    "bucket_name": "your-bucket"
}
sqs.send_message(
    QueueUrl='your-queue-url',
    MessageBody=json.dumps(job)
)


## Environment Variables Required

### Core
- `DATABASE_URL` - PostgreSQL connection string
- `GROQ_API_KEY` - For AI summarization
- `GOOGLE_API_KEY` - For embeddings
- `S3_BUCKET_NAME` - S3 bucket for files and processed data

### Queue
- `QUEUE_BACKEND` - `redis` or `sqs`
- `REDIS_HOST` - Redis hostname (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)
- `REDIS_QUEUE_NAME` - Redis queue name (default: document_jobs)
- `SQS_QUEUE_URL` - SQS queue URL

### Services
- `CHROMA_HOST` - ChromaDB hostname (default: localhost)
- `CHROMA_PORT` - ChromaDB port (default: 8000)

### Performance
- `MAX_WORKERS` - Number of parallel workers (default: 2)

## Deployment

### Docker Compose (Local)
```yaml
version: '3.8'
services:
  worker:
    image: ecs-gpu-worker:latest
    env_file:
      - .env.worker
    depends_on:
      - redis
      - postgres
      - chromadb
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: always
```

### ECS Task Definition
- Use EC2 launch type
- Attach GPU instance types (e.g., p3, p4, g4dn)
- Configure IAM role with SQS, S3, and ChromaDB permissions
- Set environment variables as above
- Use `ecs-gpu-worker:latest` image

## Monitoring

- Worker logs show job processing status
- Check queue length for backlog
- Monitor DB for file status updates
- S3 for processed JSON outputs
- ChromaDB for vector indexes
# ECS GPU Worker for Vector Trace

A long-running ECS worker that processes documents from a queue for RAG ingestion pipelines.

## Architecture

- **Queue-Driven**: Pulls jobs from Redis (local dev) or SQS (production)
- **Parallel Processing**: Uses `ProcessPoolExecutor` with configurable `MAX_WORKERS`
- **GPU-Accelerated**: Leverages NVIDIA GPUs for document processing
- **ECS EC2 Launch Type**: Designed for long-running containerized workloads

## Prerequisites

1. **Running Services**: Ensure your docker-compose services are running:
   ```bash
   cd ..
   docker-compose up -d
   ```

2. **Environment Variables**: Your `.env.worker` file should contain:
   - `DATABASE_URL` (points to postgres service)
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`
   - `S3_BUCKET_NAME`
   - `CHROMA_HOST=chromadb`
   - `CHROMA_PORT=8000`
   - `QUEUE_BACKEND=redis` (or `sqs`)
   - `REDIS_HOST=redis` (for Redis backend)
   - `SQS_QUEUE_URL=...` (for SQS backend)
   - `MAX_WORKERS=2` (default: 2)

## Queue Backends

### Redis (Local Development)
- Uses Redis lists for job queuing
- Jobs are removed on receipt (no built-in retry)
- Configure with: `QUEUE_BACKEND=redis`

### SQS (Production)
- Uses AWS SQS for reliable job queuing
- Built-in retry with visibility timeouts
- Configure with: `QUEUE_BACKEND=sqs`

## Job Format

Jobs are JSON payloads sent to the queue:

```json
{
  "project_id": 1,
  "file_id": "uuid-string",
  "s3_key": "projects/1/raw/filename.pdf",
  "original_filename": "document.pdf",
  "bucket_name": "your-s3-bucket"
}
```

## Files Overview

- `aws_gpu_worker.py` - The main ECS worker script
- `Dockerfile` - Container definition for GPU processing
- `requirements.txt` - Main Python dependencies
- `requirements-missing.txt` - Missing unstructured dependencies (separate layer)
- `test_container.sh` - Simple container test script
- `mock_backend_test.py` - End-to-end test (needs updating for queue-based workflow)

## Quick Test

```bash
# Build the worker image
docker build -t ecs-gpu-worker .

# Run with Redis backend (local dev)
docker run --rm \
  --env-file .env.worker \
  --network host \
  ecs-gpu-worker

# Or with SQS backend (production)
docker run --rm \
  --env-file .env.worker \
  ecs-gpu-worker
```

## Sending Test Jobs

### Via Redis (Local)
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)
job = {
    "project_id": 1,
    "file_id": "test-uuid",
    "s3_key": "projects/1/raw/test.pdf",
    "original_filename": "test.pdf",
    "bucket_name": "your-bucket"
}
r.lpush('document_jobs', json.dumps(job))
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
```

## Expected Output

```
🚀 Starting worker loop with 2 max workers
📡 Queue backend: redis
📬 Received 1 messages
📥 Downloading s3://your-bucket/projects/1/raw/test.pdf
📄 Partitioning document: /tmp/.../test.pdf
🔨 Creating smart chunks...
🚀 Starting LPU processing for X chunks...
📄 Exporting chunks to JSON...
✅ Exported X chunks to s3://your-bucket/projects/1/processed/test-uuid.json
🎉 Job Finished Successfully
✅ Job completed and acknowledged
```

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
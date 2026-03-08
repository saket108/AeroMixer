# Inference Contract

## Container entrypoint

The Docker image entrypoint is:

```bash
python scripts/pipeline.py --mode eval
```

## Required arguments

- `--data`: dataset source path (zip/folder/data.yaml/json)

## Recommended arguments

- `--preset prod`
- `--output-dir /work/output/infer_run`
- `--model-weight /work/output/train_run/checkpoints/model_final.pth`

## Example

```bash
docker run --rm -it \
  -v /host/datasets:/data \
  -v /host/runs:/work/output \
  aeromixer:latest \
  --data /data/aircraft_dataset \
  --preset prod \
  --output-dir /work/output/infer_run \
  --model-weight /work/output/train_run/checkpoints/model_final.pth
```

## Outputs

- `inference_manifest.json`
- `inference_summary.json`
- `inference/<dataset_name>/result_image.log`

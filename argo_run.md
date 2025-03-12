```bash
python run.py \
  --container_type docker \
  --task_config_path tasks/battleOfSexes.yaml \
  --model argo:gpt-4o \
  --per_instance_cost_limit 4.00 \
  --agent_config_path configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --max_steps 50 \
  --aliases_file ./docker/aliases.sh
```

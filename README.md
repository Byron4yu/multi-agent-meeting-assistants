#  - 多Agent智能会议助手

> 使用 LangGraph + WhisperX + FastAPI 实现的完整版本

## 技术栈

- **Agent框架**: LangGraph (状态图编排，支持并行)
- **语音转写**: WhisperX + pyannote-audio (说话人识别)
- **LLM**: MiniMax M2.7 / OpenAI GPT-4o
- **Web框架**: FastAPI + WebSocket
- **集成**: 飞书 Open API

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key
[详细配置指南](./docs/环境变量配置指南.md)

# 3. 启动服务
python -m src.main

# 4. 测试演示模式（无需音频）
curl -X POST http://localhost:8000/api/v1/meeting/demo/demo
```

## Docker 启动

```bash
docker-compose up -d
```

## 项目结构

```
python/
├── src/
│   ├── agents/                  # 5个Agent实现
│   │   ├── transcription_agent.py   # 转写Agent (WhisperX)
│   │   ├── summary_agent.py         # 摘要Agent (LLM)
│   │   ├── action_agent.py          # 待办Agent (LLM + 飞书)
│   │   ├── insight_agent.py         # 洞察Agent (规则+LLM)
│   │   └── followup_agent.py        # 跟进Agent (汇聚+推送)
│   ├── graph/
│   │   └── meeting_graph.py         # LangGraph 编排核心
│   ├── integrations/
│   │   ├── minimax_client.py        # MiniMax LLM 客户端
│   │   └── feishu_client.py         # 飞书 Open API 集成
│   ├── models/
│   │   └── schemas.py               # Pydantic 数据模型
│   ├── websocket/
│   │   └── server.py                # FastAPI + WebSocket 服务
│   └── main.py                      # 入口
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## API 文档

启动后访问: http://localhost:8000/docs

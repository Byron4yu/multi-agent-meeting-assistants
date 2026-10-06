# src/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============ 枚举定义（无依赖，放最前面） ============
class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MeetingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    BLOCKED = "blocked"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# ============ 基础数据模型（不依赖其他类） ============
class AudioConfig(BaseModel):
    sample_rate: int = 16000
    language: str = "zh"
    device: str = "cpu"
    model_size: str = "tiny"


class TranscriptionConfig(BaseModel):
    model_size: str = "tiny"
    language: str = "zh"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 16


# ============ 转录相关（注意顺序：先定义 TranscriptSegment） ============
class TranscriptSegment(BaseModel):
    speaker: str = Field(..., description="说话人标签")
    text: str = Field(..., description="转写文本")
    start: float = Field(..., description="开始时间（秒）")
    end: float = Field(..., description="结束时间（秒）")
    confidence: Optional[float] = None


class TranscriptResult(BaseModel):
    """转录结果（TranscriptionAgent 输出）"""
    segments: List[TranscriptSegment] = []
    full_text: str = ""
    language: Optional[str] = "zh"
    duration: Optional[float] = None


# 为了兼容旧代码，提供别名（TranscriptionResult 已存在，但有些地方用 TranscriptResult）
TranscriptionResult = TranscriptResult  # 使得两个名字都能用


# ============ 摘要相关 ============
class SummaryPoint(BaseModel):
    topic: str
    content: str
    importance: Optional[int] = 1


class TopicSummary(BaseModel):
    topic: str = Field(..., description="议题标题")
    content: str = Field(..., description="议题内容或讨论要点")
    conclusion: Optional[str] = Field(None, description="议题结论")
    participants: List[str] = Field(default_factory=list, description="参与讨论的说话人")
    discussion_points: List[str] = Field(default_factory=list, description="讨论要点（备选字段）")
    
    class Config:
        populate_by_name = True
    
    def __init__(self, **data):
        # 兼容处理：如果传入了 title，映射到 topic
        if "title" in data and "topic" not in data:
            data["topic"] = data.pop("title")
        
        # 兼容处理：如果传入了 description 或 discussion_points，映射到 content
        if "description" in data and "content" not in data:
            data["content"] = data.pop("description")
        if "discussion_points" in data and "content" not in data:
            # 把 discussion_points 列表转为字符串，作为 content
            points = data.pop("discussion_points")
            if isinstance(points, list):
                data["content"] = "；".join(points)
            else:
                data["content"] = str(points)
        
        super().__init__(**data)


class MeetingSummary(BaseModel):
    title: str = Field(..., description="会议标题")
    date: str = Field(default="", description="会议日期")
    participants: List[str] = Field(default_factory=list, description="参会人列表")
    topics: List[TopicSummary] = Field(default_factory=list, description="议题列表")
    objective: Optional[str] = None
    key_points: List[SummaryPoint] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    discussion_summary: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)


# ============ 洞察相关 ============
class SpeakerStats(BaseModel):
    speaker: str = Field(..., description="说话人标签")
    speaking_duration: float = Field(0.0, description="发言总时长（秒）")
    speaking_ratio: float = Field(0.0, description="发言占比（0-1）")
    word_count: int = Field(0, description="总字数")
    segment_count: int = Field(0, description="发言片段数量")


class MeetingInsight(BaseModel):
    meeting_id: str = Field(..., description="会议ID")
    overall_sentiment: SentimentType = Field(SentimentType.NEUTRAL, description="整体情感倾向")
    sentiment_score: float = Field(0.5, description="情感得分 (0-1)")
    speaker_stats: List[SpeakerStats] = Field(default_factory=list, description="发言统计")
    efficiency_score: float = Field(5.0, description="效率评分 (0-10)")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    highlights: List[str] = Field(default_factory=list, description="会议亮点")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


# ============ 行动项相关 ============
class ActionItem(BaseModel):
    id: Optional[str] = None
    description: str = ""
    task: str = ""
    assignee: str = "未指定"
    due_date: Optional[str] = None
    deadline: str = ""
    priority: Priority = Priority.MEDIUM
    status: ActionStatus = ActionStatus.PENDING
    source_segment: Optional[str] = None
    context: str = ""
    feishu_task_id: Optional[str] = None


class ActionResult(BaseModel):
    success: bool = True
    meeting_id: str = "unknown"
    action_items: List["ActionItem"] = Field(default_factory=list)
    sync_status: Dict[str, str] = Field(default_factory=dict)
    action_id: Optional[str] = None
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============ 跟进任务相关 ============
class FollowUpTask(BaseModel):
    task_id: Optional[str] = None
    description: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    source_action: Optional[str] = None
    status: str = "pending"


class FollowUpResult(BaseModel):
    meeting_id: str = "unknown"
    success: bool = True
    summary_sent: bool = False
    recipients: List[str] = Field(default_factory=list)
    feishu_tasks_created: List[str] = Field(default_factory=list)
    reminders_scheduled: int = 0
    report_url: Optional[str] = None
    task_id: Optional[str] = None
    integration_type: Optional[str] = None
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============ 最终输出结果 ============
class MeetingResult(BaseModel):
    meeting_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    transcription: Optional[TranscriptResult] = None
    summary: Optional[MeetingSummary] = None
    action_items: List[ActionItem] = Field(default_factory=list)
    insights: List[MeetingInsight] = Field(default_factory=list)
    follow_up_tasks: List[FollowUpTask] = Field(default_factory=list)
    processing_time: Optional[float] = None
    agent_traces: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============ API 请求/响应 ============
class MeetingRequest(BaseModel):
    audio_file_path: Optional[str] = None
    audio_base64: Optional[str] = None
    config: Optional[AudioConfig] = None
    extra_context: Optional[Dict[str, Any]] = None


class MeetingResponse(BaseModel):
    status: str = "success"
    data: Optional[MeetingResult] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None


# ============ LangGraph 状态 ============
class MeetingState(BaseModel):
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    language: str = "zh"
    transcription: Optional[TranscriptResult] = None
    summary: Optional[MeetingSummary] = None
    action_items: List[ActionItem] = Field(default_factory=list)
    insights: List[MeetingInsight] = Field(default_factory=list)
    follow_up_tasks: List[FollowUpTask] = Field(default_factory=list)
    status: MeetingStatus = MeetingStatus.PENDING
    current_agent: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    meeting_id: Optional[str] = None
    trace: Dict[str, Any] = Field(default_factory=dict)


# ============ 辅助函数 ============
def create_initial_state(audio_path: Optional[str] = None, audio_base64: Optional[str] = None, **kwargs) -> MeetingState:
    return MeetingState(
        audio_path=audio_path,
        audio_base64=audio_base64,
        status=MeetingStatus.PENDING,
        started_at=datetime.now(),
        **kwargs
    )
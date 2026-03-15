# Video Rater — Taste Prediction Database

一个基于 Streamlit 的视频标注 Web 工具，用于人工审阅视频、查看 AI 生成的元数据、打分并将偏好数据存入 PostgreSQL，构建「口味预测」数字孪生数据库。

## 功能概览

- **视频播放**：自动循环播放，自适应视口（PC / 手机均无需滚动）
- **AI 元数据展示**：读取与每个视频对应的 `_analysis.json`，展示标签、色板、描述等字段
- **特征标签提取**：从帧描述和摘要文本中提取 2-6 字的特征短语，支持两种方式：
  - **云端 LLM**（可选）：调用 OpenAI / OpenRouter 等兼容 API，语义质量更高
  - **本地规则**（默认回退）：正则提取中文短语，无需网络
- **打分与标注**：1–10 分滑杆 + 多选偏好特征标签，提交后写入 PostgreSQL
- **进度追踪**：顶部实时显示已评 / 剩余 / 总数；上下视频自由切换
- **LLM 调试面板**：折叠展示配置状态、提取结果、原始响应，支持一键测试 API 连接

## 目录结构要求

运行前请将工作目录（或 `VIDEO_RATER_BASE`）设置为同时包含 `1/`（视频）和 `2/`（分析）的父目录：

```
<base>/
├── 1/                          # 视频文件目录
│   └── <subfolder>/
│       └── <name>.mp4
└── 2/                          # 分析 JSON 目录（结构与 1/ 镜像）
    └── <subfolder>/
        └── <name>_analysis.json
```

> `<name>_analysis.json` 中需包含描述性文本字段（如 `summary`、`description`、`tags`、`color_palette` 等）。

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入数据库连接信息和（可选）LLM API 密钥
```

### 3. 准备 PostgreSQL

```sql
CREATE DATABASE video_rater;
-- 表结构由应用首次启动时自动创建
```

### 4. 启动应用

```bash
# 方式一：直接用 python 启动（推荐，自动拉起 streamlit）
python app.py

# 方式二：直接调用 streamlit
VIDEO_RATER_STREAMLIT=1 streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `PGHOST` | `localhost` | PostgreSQL 主机 |
| `PGPORT` | `5432` | PostgreSQL 端口 |
| `PGDATABASE` | `video_rater` | 数据库名 |
| `PGUSER` | `postgres` | 数据库用户 |
| `PGPASSWORD` | _(空)_ | 数据库密码 |
| `VIDEO_RATER_BASE` | `.`（当前目录）| 视频与分析文件的父目录路径 |
| `VIDEO_RATER_AUTH_ENABLED` | `false` | 设为 `true`/`1`/`yes` 启用登录认证（公网部署建议开启） |
| `VIDEO_RATER_ADMIN_USER` | `admin` | 首次创建的管理员用户名（仅当数据库尚无用户时生效） |
| `VIDEO_RATER_ADMIN_PASSWORD` | _(空)_ | 管理员初始密码；启用认证且数据库无用户时自动创建该账号 |
| `VIDEO_RATER_OIDC_ISSUER` | _(未设置)_ | OIDC Issuer（如 Casdoor 地址），如 `https://door.casdoor.com` |
| `VIDEO_RATER_OIDC_CLIENT_ID` | _(未设置)_ | OIDC 应用 Client ID |
| `VIDEO_RATER_OIDC_CLIENT_SECRET` | _(未设置)_ | OIDC 应用 Client Secret（可选，视 IdP 要求） |
| `VIDEO_RATER_OIDC_REDIRECT_URI` | _(未设置)_ | 回调地址，须与 Casdoor 应用内 Redirect URLs 一致 |
| `VIDEO_RATER_OIDC_SCOPE` | `openid profile email` | OIDC scope |
| `LLM_APP_URL` | _(未设置)_ | LLM API base URL，如 `https://api.openai.com/v1` |
| `LLM_API_KEY` | _(未设置)_ | LLM API 密钥 |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | 使用的模型名称 |

> **注意**：修改 `.env` 后需重启应用才能生效。

### 身份认证（公网部署）

部署到公网时建议开启认证，避免未授权访问：

1. 在 `.env` 中设置 `VIDEO_RATER_AUTH_ENABLED=true`，并设置 `VIDEO_RATER_ADMIN_USER` 与 `VIDEO_RATER_ADMIN_PASSWORD`。
2. 首次启动时，若数据库中没有用户，会自动创建该管理员账号。
3. 用户表 `video_rater_users` 与 `video_preferences` 一样由应用自动创建；后续如需新增用户，需在数据库中手动插入（密码需用 bcrypt 哈希）。

登录后可在侧边栏看到当前用户名并点击「退出登录」。

### OIDC / Casdoor 登录

在启用认证的前提下，配置 OIDC 后登录页会显示「使用 Casdoor 登录」：

1. 在 Casdoor 控制台创建应用，记下 **Client ID**、**Client Secret**；在应用的 **Redirect URLs** 中加入本应用的回调地址（例如 `https://你的域名/` 或 `http://localhost:8501/`）。
2. 在 `.env` 中设置：
   - `VIDEO_RATER_OIDC_ISSUER`：Casdoor 地址（如 `https://door.casdoor.com`，不要末尾斜杠）
   - `VIDEO_RATER_OIDC_CLIENT_ID`、`VIDEO_RATER_OIDC_CLIENT_SECRET`
   - `VIDEO_RATER_OIDC_REDIRECT_URI`：与 Casdoor 中填写的回调地址完全一致
3. 重启应用，在登录页点击「使用 Casdoor 登录」完成 OIDC 流程。登录成功后用户名来自 IdP 的 `preferred_username`、`sub` 或 `name`。

本地账号密码登录与 OIDC 可同时使用。

## LLM 配置示例

**OpenAI**

```env
LLM_APP_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini
```

**OpenRouter**

```env
LLM_APP_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL_NAME=openai/gpt-4o-mini
```

不配置 LLM 时，应用自动使用本地正则规则提取特征短语，不影响核心评分功能。

## 数据库表结构

应用启动时自动创建以下表。

**video_preferences**（标注数据）：

```sql
CREATE TABLE video_preferences (
    id            SERIAL PRIMARY KEY,
    video_path    VARCHAR(512) UNIQUE NOT NULL,  -- 相对路径（1/<subfolder>/<name>.mp4）
    json_path     VARCHAR(512) NOT NULL,          -- 相对路径（2/<subfolder>/<name>_analysis.json）
    overall_score NUMERIC(4, 2) NOT NULL,         -- 综合评分 1.0–10.0
    liked_features JSONB,                         -- 选中的偏好特征标签列表
    raw_analysis  JSONB NOT NULL,                 -- 原始 analysis.json 全量数据
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

**video_rater_users**（启用认证时创建，用于登录）：

```sql
CREATE TABLE video_rater_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

## 依赖

| 包 | 最低版本 | 说明 |
|---|---|---|
| `streamlit` | 1.37.0 | Web UI 框架（需 `st.fragment` 支持） |
| `psycopg2-binary` | 2.9.9 | PostgreSQL 驱动 |
| `python-dotenv` | 1.0.0 | 读取 `.env` 配置文件 |
| `openai` | 1.0.0 | LLM 特征提取（可选） |
| `bcrypt` | 4.0.0 | 密码哈希（启用认证时使用） |
| `httpx` | 0.27.x | OIDC 发现、token 与 userinfo 请求 |

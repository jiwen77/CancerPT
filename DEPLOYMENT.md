# 部署指南 (Deployment Guide)

## 文件清单 (File Checklist)

部署前确保以下文件已准备好：

- [x] `app.py` - 主应用文件
- [x] `sample_data.py` - 数据处理和模型文件
- [x] `requirements.txt` - Python依赖包
- [x] `README.md` - 项目说明文档
- [x] `.streamlit/config.toml` - Streamlit配置文件

## 部署步骤 (Deployment Steps)

### 方法一：GitHub + Streamlit Community Cloud

#### 1. 上传到GitHub
1. 访问 [GitHub.com](https://github.com)
2. 创建新仓库 "head-neck-cancer-survival-app"
3. 设置为Public（公开）
4. 上传所有文件：
   - 拖拽文件到GitHub界面
   - 或使用"uploading an existing file"链接

#### 2. 部署到Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择仓库：`yourusername/head-neck-cancer-survival-app`
5. 分支：`main`
6. 主文件：`app.py`
7. 点击 "Deploy!"

#### 3. 等待部署完成
- 首次部署需要5-10分钟
- Streamlit会自动安装dependencies
- 部署成功后会获得公开URL

### 方法二：本地压缩包上传

#### 1. 创建压缩包
将以下文件打包为ZIP：
```
head-neck-cancer-survival-app/
├── app.py
├── sample_data.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

#### 2. 上传到GitHub
1. 在GitHub创建新仓库
2. 选择 "uploading an existing file"
3. 上传ZIP文件并解压

#### 3. 按方法一第2步部署

## 注意事项 (Important Notes)

### 依赖管理
- 确保`requirements.txt`包含所有必要的包
- 版本号要兼容（当前版本已测试）

### 文件大小
- Streamlit Cloud限制单个文件<100MB
- 总仓库大小<1GB

### 数据文件
- 模型文件会在首次运行时自动生成
- 不需要预先上传训练好的模型

### 性能优化
- 使用`@st.cache_resource`缓存模型
- 避免在每次交互时重新训练模型

## 故障排除 (Troubleshooting)

### 常见问题

1. **依赖安装失败**
   - 检查requirements.txt格式
   - 确保包名和版本号正确

2. **内存不足**
   - Streamlit Cloud免费版限制1GB内存
   - 优化数据处理代码

3. **部署超时**
   - 减少启动时的计算量
   - 使用缓存机制

4. **权限问题**
   - 确保GitHub仓库是Public
   - 检查Streamlit Cloud访问权限

## 部署后测试 (Post-Deployment Testing)

部署成功后测试以下功能：

- [ ] 页面正常加载
- [ ] 患者数据输入功能
- [ ] 生存率计算和显示
- [ ] 图表可视化
- [ ] 患者档案保存/加载
- [ ] 比较功能
- [ ] 风险因子分析
- [ ] 数据上传功能

## 更新部署 (Updating Deployment)

修改代码后：
1. 将新代码推送到GitHub仓库
2. Streamlit Cloud会自动检测更新
3. 自动重新部署（通常1-2分钟）

## 访问控制 (Access Control)

- 免费版应用是公开的
- 如需私有部署，考虑升级到付费版
- 或使用其他部署平台（Heroku, AWS等）

## 成本 (Cost)

- Streamlit Community Cloud：免费
- 限制：公开应用，有资源限制
- 适合：演示、原型、小规模应用

---

**部署完成后，你的应用将可以通过类似以下URL访问：**
`https://yourusername-head-neck-cancer-survival-app-app-xxx.streamlit.app/` 
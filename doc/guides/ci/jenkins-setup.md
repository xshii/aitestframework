# Jenkins 质量防线搭建指南

## 概述

本文档介绍如何在Linux环境下搭建基于Jenkins的CI/CD质量防线，实现：
- 代码提交自动触发构建
- 代码质量检查（pylint/ruff）
- 单元测试和覆盖率
- 集成测试
- 质量门禁（不达标则阻断合入）

## 1. Jenkins安装

### 1.1 系统要求

- OS: Ubuntu 20.04+ / CentOS 7+
- Java: OpenJDK 11+
- Python: 3.9+
- 内存: 4GB+
- 磁盘: 50GB+

### 1.2 安装Java

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y openjdk-11-jdk

# CentOS/RHEL
sudo yum install -y java-11-openjdk java-11-openjdk-devel

# 验证
java -version
```

### 1.3 安装Jenkins

```bash
# Ubuntu/Debian
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

sudo apt update
sudo apt install -y jenkins

# CentOS/RHEL
sudo wget -O /etc/yum.repos.d/jenkins.repo \
    https://pkg.jenkins.io/redhat-stable/jenkins.repo
sudo rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key
sudo yum install -y jenkins

# 启动服务
sudo systemctl start jenkins
sudo systemctl enable jenkins

# 查看初始密码
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

### 1.4 访问Jenkins

1. 浏览器访问: `http://<服务器IP>:8080`
2. 输入初始密码
3. 安装推荐插件
4. 创建管理员账户

## 2. Jenkins插件配置

### 2.1 必需插件

在 **Manage Jenkins → Plugins → Available plugins** 安装：

| 插件名 | 用途 |
|-------|------|
| Git | Git仓库集成 |
| Pipeline | 流水线支持 |
| Pipeline: Stage View | 流水线可视化 |
| Blue Ocean | 现代化UI |
| Cobertura | 覆盖率报告 |
| JUnit | 测试报告 |
| Warnings Next Generation | 代码检查报告 |
| HTML Publisher | HTML报告发布 |
| GitLab / GitHub | 代码托管平台集成 |
| Credentials Binding | 凭据管理 |

### 2.2 Python环境配置

在Jenkins服务器上安装Python环境：

```bash
# 安装Python和pip
sudo apt install -y python3.9 python3.9-venv python3-pip

# 创建全局虚拟环境（可选）
sudo mkdir -p /opt/jenkins-python
sudo python3.9 -m venv /opt/jenkins-python/venv
sudo /opt/jenkins-python/venv/bin/pip install --upgrade pip

# 安装常用工具
sudo /opt/jenkins-python/venv/bin/pip install \
    pytest pytest-cov pytest-html \
    pylint ruff \
    numpy pyyaml
```

## 3. 创建流水线任务

### 3.1 新建Pipeline任务

1. 点击 **New Item**
2. 输入名称: `aitestframework`
3. 选择 **Pipeline**
4. 点击 **OK**

### 3.2 配置Pipeline

在任务配置页面：

**General**:
- 勾选 **Discard old builds**
  - Days to keep builds: 30
  - Max # of builds to keep: 50

**Build Triggers**:
- 勾选 **Poll SCM** (或配置Webhook)
  - Schedule: `H/5 * * * *` (每5分钟检查)

**Pipeline**:
- Definition: **Pipeline script from SCM**
- SCM: **Git**
- Repository URL: `<仓库地址>`
- Credentials: 选择或添加凭据
- Branch: `*/main` 或 `*/*`
- Script Path: `Jenkinsfile`

### 3.3 Webhook配置（推荐）

**GitLab**:
1. 项目设置 → Webhooks
2. URL: `http://<jenkins>/project/aitestframework`
3. Trigger: Push events, Merge request events

**GitHub**:
1. 仓库设置 → Webhooks → Add webhook
2. Payload URL: `http://<jenkins>/github-webhook/`
3. Content type: `application/json`
4. Events: Push, Pull request

## 4. 质量门禁规则

### 4.1 门禁指标

| 指标 | 阈值 | 说明 |
|-----|------|------|
| 单元测试通过率 | 100% | 所有测试必须通过 |
| 代码覆盖率 | ≥80% | 行覆盖率 |
| Pylint评分 | ≥8.0 | 代码质量评分 |
| Ruff检查 | 0 errors | 无错误 |
| 集成测试 | 100% | 关键流程测试 |

### 4.2 门禁实现

在Jenkinsfile中通过以下方式实现门禁：

```groovy
// 测试通过率检查
if (testResult.failCount > 0) {
    error "Tests failed: ${testResult.failCount} failures"
}

// 覆盖率检查
if (coverage < 80) {
    error "Coverage ${coverage}% is below threshold 80%"
}

// Pylint评分检查
if (pylintScore < 8.0) {
    error "Pylint score ${pylintScore} is below threshold 8.0"
}
```

## 5. 多分支流水线（可选）

对于需要支持多分支的项目：

1. 新建 **Multibranch Pipeline**
2. 配置Branch Sources为Git仓库
3. 配置Behaviors:
   - Discover branches
   - Discover pull requests
4. 所有包含Jenkinsfile的分支都会自动构建

## 6. 凭据管理

### 6.1 添加Git凭据

1. **Manage Jenkins → Credentials → System → Global credentials**
2. 点击 **Add Credentials**
3. Kind: **Username with password** 或 **SSH Username with private key**
4. 填写用户名/密码或SSH私钥

### 6.2 添加其他凭据

可根据需要添加：
- NPU设备访问凭据
- Docker Registry凭据
- 通知服务Token

## 7. 通知配置

### 7.1 邮件通知

**Manage Jenkins → System → E-mail Notification**:
- SMTP server: `smtp.example.com`
- 配置发件人邮箱

### 7.2 企业微信/钉钉通知

安装对应插件后，在Jenkinsfile中添加：

```groovy
post {
    failure {
        // 企业微信通知
        sh '''
            curl -X POST "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx" \
                -H "Content-Type: application/json" \
                -d '{"msgtype":"text","text":{"content":"构建失败: ${JOB_NAME} #${BUILD_NUMBER}"}}'
        '''
    }
}
```

## 8. 常见问题

### Q: Jenkins无法拉取代码
A: 检查凭据配置，确保Jenkins用户有仓库访问权限

### Q: Python环境找不到
A: 在Jenkinsfile中指定Python路径或使用虚拟环境

### Q: 测试超时
A: 在Jenkinsfile中增加timeout配置

### Q: 权限不足
A: 检查Jenkins用户权限，可能需要将jenkins用户加入docker组等

## 9. 参考资料

- [Jenkins官方文档](https://www.jenkins.io/doc/)
- [Pipeline语法](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [Blue Ocean文档](https://www.jenkins.io/doc/book/blueocean/)

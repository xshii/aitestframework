#!/usr/bin/env groovy
/**
 * AI Test Framework - Jenkins Pipeline
 *
 * 质量防线流水线，包含：
 * 1. 代码检查 (pylint, ruff)
 * 2. 单元测试 + 覆盖率
 * 3. 集成测试
 * 4. 质量门禁
 */

pipeline {
    agent any

    options {
        // 构建超时
        timeout(time: 30, unit: 'MINUTES')
        // 保留构建历史
        buildDiscarder(logRotator(numToKeepStr: '50', daysToKeepStr: '30'))
        // 禁止并发构建
        disableConcurrentBuilds()
        // 时间戳
        timestamps()
    }

    environment {
        // Python路径
        PYTHON = 'python3'
        PIP = 'pip3'
        // 项目路径
        PYTHONPATH = "${WORKSPACE}:${WORKSPACE}/libs"
        // 质量门禁阈值
        COVERAGE_THRESHOLD = '80'
        PYLINT_THRESHOLD = '8.0'
    }

    parameters {
        booleanParam(
            name: 'SKIP_LINT',
            defaultValue: false,
            description: '跳过代码检查'
        )
        booleanParam(
            name: 'SKIP_INTEGRATION_TESTS',
            defaultValue: false,
            description: '跳过集成测试'
        )
        booleanParam(
            name: 'FORCE_BUILD',
            defaultValue: false,
            description: '强制构建（忽略质量门禁）'
        )
    }

    stages {
        stage('Prepare') {
            steps {
                echo "========== 环境准备 =========="
                sh '''
                    echo "Python版本: $(${PYTHON} --version)"
                    echo "工作目录: ${WORKSPACE}"
                    echo "分支: ${GIT_BRANCH:-unknown}"
                    echo "提交: ${GIT_COMMIT:-unknown}"
                '''

                // 创建虚拟环境
                sh '''
                    ${PYTHON} -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements/dev.txt
                '''
            }
        }

        stage('Lint') {
            when {
                expression { !params.SKIP_LINT }
            }
            parallel {
                stage('Pylint') {
                    steps {
                        echo "========== Pylint 代码检查 =========="
                        sh '''
                            . .venv/bin/activate
                            pylint aitestframework --output-format=parseable \
                                --reports=y \
                                --exit-zero \
                                > pylint-report.txt 2>&1 || true

                            # 提取评分
                            SCORE=$(grep "Your code has been rated" pylint-report.txt | \
                                sed 's/.*rated at \\([0-9.]*\\).*/\\1/' || echo "0")
                            echo "Pylint Score: ${SCORE}"
                            echo "${SCORE}" > pylint-score.txt
                        '''
                    }
                    post {
                        always {
                            // 发布Pylint报告
                            recordIssues(
                                tools: [pyLint(pattern: 'pylint-report.txt')],
                                qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]]
                            )
                        }
                    }
                }

                stage('Ruff') {
                    steps {
                        echo "========== Ruff 快速检查 =========="
                        sh '''
                            . .venv/bin/activate
                            ruff check aitestframework libs/aidevtools \
                                --output-format=json \
                                --exit-zero \
                                > ruff-report.json 2>&1 || true

                            # 统计错误数
                            ERROR_COUNT=$(cat ruff-report.json | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" || echo "0")
                            echo "Ruff Errors: ${ERROR_COUNT}"
                            echo "${ERROR_COUNT}" > ruff-errors.txt
                        '''
                    }
                }
            }
        }

        stage('Unit Tests') {
            steps {
                echo "========== 单元测试 =========="
                sh '''
                    . .venv/bin/activate
                    pytest aitestframework/ \
                        --junitxml=test-results/unit-tests.xml \
                        --cov=aitestframework \
                        --cov-report=xml:coverage/coverage.xml \
                        --cov-report=html:coverage/html \
                        --cov-fail-under=0 \
                        -v || true
                '''
            }
            post {
                always {
                    // 发布测试报告
                    junit allowEmptyResults: true, testResults: 'test-results/unit-tests.xml'
                    // 发布覆盖率报告
                    publishHTML(target: [
                        allowMissing: true,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'coverage/html',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }

        stage('Integration Tests') {
            when {
                expression { !params.SKIP_INTEGRATION_TESTS }
            }
            steps {
                echo "========== 集成测试 =========="
                sh '''
                    . .venv/bin/activate
                    pytest tests/it tests/st \
                        --junitxml=test-results/integration-tests.xml \
                        -v || true
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results/integration-tests.xml'
                }
            }
        }

        stage('Quality Gate') {
            when {
                expression { !params.FORCE_BUILD }
            }
            steps {
                echo "========== 质量门禁检查 =========="
                script {
                    def gatesPassed = true
                    def gateResults = []

                    // 1. 检查单元测试结果
                    def testResult = currentBuild.rawBuild.getAction(hudson.tasks.junit.TestResultAction.class)
                    if (testResult != null) {
                        def failCount = testResult.getFailCount()
                        def totalCount = testResult.getTotalCount()
                        def passRate = totalCount > 0 ? ((totalCount - failCount) * 100 / totalCount) : 0

                        if (failCount > 0) {
                            gatesPassed = false
                            gateResults.add("❌ 单元测试: ${failCount}/${totalCount} 失败")
                        } else {
                            gateResults.add("✅ 单元测试: ${totalCount} 全部通过")
                        }
                    }

                    // 2. 检查覆盖率
                    if (fileExists('coverage/coverage.xml')) {
                        def coverageXml = readFile('coverage/coverage.xml')
                        def matcher = coverageXml =~ /line-rate="([0-9.]+)"/
                        if (matcher.find()) {
                            def coverage = (matcher.group(1) as Double) * 100
                            def threshold = env.COVERAGE_THRESHOLD as Double

                            if (coverage < threshold) {
                                gatesPassed = false
                                gateResults.add("❌ 覆盖率: ${coverage.round(1)}% < ${threshold}%")
                            } else {
                                gateResults.add("✅ 覆盖率: ${coverage.round(1)}% ≥ ${threshold}%")
                            }
                        }
                    }

                    // 3. 检查Pylint评分
                    if (fileExists('pylint-score.txt')) {
                        def pylintScore = readFile('pylint-score.txt').trim() as Double
                        def threshold = env.PYLINT_THRESHOLD as Double

                        if (pylintScore < threshold) {
                            gatesPassed = false
                            gateResults.add("❌ Pylint: ${pylintScore} < ${threshold}")
                        } else {
                            gateResults.add("✅ Pylint: ${pylintScore} ≥ ${threshold}")
                        }
                    }

                    // 4. 检查Ruff错误
                    if (fileExists('ruff-errors.txt')) {
                        def ruffErrors = readFile('ruff-errors.txt').trim() as Integer

                        if (ruffErrors > 0) {
                            // Ruff错误暂时只警告，不阻断
                            gateResults.add("⚠️ Ruff: ${ruffErrors} 个问题")
                        } else {
                            gateResults.add("✅ Ruff: 无问题")
                        }
                    }

                    // 输出门禁结果
                    echo "========== 质量门禁结果 =========="
                    gateResults.each { echo it }

                    if (!gatesPassed) {
                        error "质量门禁未通过，请修复上述问题后重新提交"
                    }
                }
            }
        }

        stage('Archive') {
            steps {
                echo "========== 归档产物 =========="
                // 归档测试报告
                archiveArtifacts artifacts: 'test-results/**/*.xml', allowEmptyArchive: true
                archiveArtifacts artifacts: 'coverage/**/*', allowEmptyArchive: true
                archiveArtifacts artifacts: 'pylint-report.txt', allowEmptyArchive: true
                archiveArtifacts artifacts: 'ruff-report.json', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            echo "========== 清理工作区 =========="
            // 清理虚拟环境（可选）
            // sh 'rm -rf .venv'

            // 清理临时文件
            sh 'rm -rf __pycache__ .pytest_cache *.pyc'
        }

        success {
            echo "✅ 构建成功!"
            // 可添加成功通知
        }

        failure {
            echo "❌ 构建失败!"
            // 可添加失败通知
            // emailext(
            //     subject: "构建失败: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
            //     body: "详情: ${env.BUILD_URL}",
            //     to: "dev-team@example.com"
            // )
        }

        unstable {
            echo "⚠️ 构建不稳定（测试或质量检查有问题）"
        }
    }
}

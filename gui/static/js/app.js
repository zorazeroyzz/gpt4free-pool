/**
 * FreeAPI Pool - 前端交互逻辑
 * 免费API流量池管理界面
 */

const API_BASE = window.location.origin;

// ============ 页面切换 ============

function switchPage(page) {
    // 更新导航高亮
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');

    // 切换页面
    document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
    document.getElementById(`page-${page}`)?.classList.add('active');

    // 更新标题
    const titles = {
        dashboard: '控制面板',
        providers: '提供商管理',
        models: '模型列表',
        playground: '对话测试',
        apikeys: 'API密钥',
        docs: '接口文档',
    };
    document.getElementById('pageTitle').textContent = titles[page] || page;

    // 加载页面数据
    if (page === 'dashboard') refreshDashboard();
    if (page === 'providers') loadProviders();
    if (page === 'models') loadModels();
    if (page === 'playground') loadPlaygroundOptions();
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// ============ 控制面板 ============

async function refreshDashboard() {
    try {
        const [statsRes, providersRes] = await Promise.all([
            fetch(`${API_BASE}/api/stats`),
            fetch(`${API_BASE}/api/providers`),
        ]);

        const stats = await statsRes.json();
        const providersData = await providersRes.json();

        // 更新统计数字
        document.getElementById('stat-requests').textContent = stats.total_requests.toLocaleString();
        document.getElementById('stat-success').textContent = stats.success_rate;
        document.getElementById('stat-models').textContent = stats.total_models;
        document.getElementById('stat-providers').textContent = `${stats.active_providers}/${stats.total_providers}`;

        // 渲染提供商概览卡片
        const overview = document.getElementById('providerOverview');
        overview.innerHTML = providersData.providers.map(p => `
            <div class="provider-card">
                <div class="provider-card-header">
                    <span class="provider-card-name">${p.label || p.name}</span>
                    <span class="badge ${p.working ? 'badge-success' : 'badge-error'}">${p.working ? '运行中' : '已停止'}</span>
                </div>
                <div class="provider-card-stats">
                    <span>模型: ${p.models?.length || 0}</span>
                    <span>请求: ${p.stats?.total_requests || 0}</span>
                    <span>成功率: ${p.stats?.success_rate || '0.0%'}</span>
                </div>
            </div>
        `).join('');

    } catch (e) {
        console.error('刷新控制面板失败:', e);
    }
}

// ============ 提供商管理 ============

async function loadProviders() {
    try {
        const res = await fetch(`${API_BASE}/api/providers`);
        const data = await res.json();

        const tbody = document.getElementById('providersTableBody');
        tbody.innerHTML = data.providers.map(p => `
            <tr>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <strong>${p.name}</strong>
                        ${p.url ? `<a href="${p.url}" target="_blank" style="color:var(--text-muted);font-size:11px;">↗</a>` : ''}
                    </div>
                    <div style="font-size:11px;color:var(--text-muted);">${p.label || ''}</div>
                </td>
                <td><span class="badge ${p.working ? 'badge-success' : 'badge-error'}">${p.working ? '运行中' : '已停止'}</span></td>
                <td>${p.models?.length || 0}</td>
                <td>${p.stats?.total_requests || 0}</td>
                <td>${p.stats?.success_rate || '0.0%'}</td>
                <td>${p.stats?.avg_latency_ms || '0'}ms</td>
                <td>${p.needs_auth ? '<span class="badge badge-warning">需要</span>' : '<span class="badge badge-info">免费</span>'}</td>
                <td>
                    <div style="display:flex;gap:6px;">
                        <button class="btn btn-sm" onclick="toggleProvider('${p.name}')">${p.working ? '禁用' : '启用'}</button>
                        <button class="btn btn-sm" onclick="checkHealth('${p.name}')">检测</button>
                    </div>
                </td>
            </tr>
        `).join('');

    } catch (e) {
        console.error('加载提供商失败:', e);
    }
}

async function toggleProvider(name) {
    try {
        const res = await fetch(`${API_BASE}/api/providers/${name}/toggle`, { method: 'POST' });
        const data = await res.json();
        showToast(data.message, 'success');
        loadProviders();
    } catch (e) {
        showToast('操作失败: ' + e.message, 'error');
    }
}

async function checkHealth(name) {
    showToast(`正在检测 ${name}...`, 'info');
    try {
        const res = await fetch(`${API_BASE}/api/providers/${name}/health`, { method: 'POST' });
        const data = await res.json();
        showToast(
            `${name}: ${data.healthy ? '✓ 服务正常' : '✗ 服务异常'}`,
            data.healthy ? 'success' : 'error'
        );
        loadProviders();
    } catch (e) {
        showToast(`检测失败: ${e.message}`, 'error');
    }
}

async function checkAllHealth() {
    showToast('正在检测所有提供商...', 'info');
    try {
        const res = await fetch(`${API_BASE}/api/providers`);
        const data = await res.json();
        for (const p of data.providers) {
            await checkHealth(p.name);
        }
    } catch (e) {
        showToast('检测失败', 'error');
    }
}

// ============ 模型列表 ============

let allModels = [];

async function loadModels() {
    try {
        const res = await fetch(`${API_BASE}/api/models`);
        const data = await res.json();
        allModels = data.models || [];
        renderModels(allModels);
    } catch (e) {
        console.error('加载模型失败:', e);
    }
}

function filterModels(category) {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    if (category === 'all') {
        renderModels(allModels);
    } else {
        renderModels(allModels.filter(m => m.category === category));
    }
}

function renderModels(models) {
    const grid = document.getElementById('modelsGrid');
    if (models.length === 0) {
        grid.innerHTML = '<div class="loading">暂无模型</div>';
        return;
    }

    const catLabels = { chat: '对话', code: '代码', reasoning: '推理', vision: '视觉', image: '图像', audio: '音频' };
    const catClasses = { chat: 'cat-chat', code: 'cat-code', reasoning: 'cat-reasoning', vision: 'cat-vision', image: 'cat-image', audio: 'cat-audio' };

    grid.innerHTML = models.map(m => `
        <div class="model-card" data-category="${m.category}">
            <div class="model-card-header">
                <span class="model-name">${m.id}</span>
                <span class="model-category ${catClasses[m.category] || 'cat-chat'}">${catLabels[m.category] || m.category}</span>
            </div>
            <div class="model-provider">
                <strong>${m.label}</strong> · ${m.base_provider}
            </div>
            <div class="model-providers-list">
                ${(m.providers || []).map(p => `<span class="provider-tag">${p}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

// ============ 对话测试 ============

let chatHistory = [];
let isStreaming = false;

async function loadPlaygroundOptions() {
    try {
        const [modelsRes, providersRes] = await Promise.all([
            fetch(`${API_BASE}/api/models`),
            fetch(`${API_BASE}/api/providers`),
        ]);

        const modelsData = await modelsRes.json();
        const providersData = await providersRes.json();

        // 填充模型下拉
        const modelSelect = document.getElementById('playgroundModel');
        modelSelect.innerHTML = (modelsData.models || []).map(m =>
            `<option value="${m.id}">${m.label} (${m.id})</option>`
        ).join('');

        // 填充提供商下拉
        const providerSelect = document.getElementById('playgroundProvider');
        providerSelect.innerHTML = '<option value="">自动选择</option>' +
            (providersData.providers || []).filter(p => p.working).map(p =>
                `<option value="${p.name}">${p.label || p.name}</option>`
            ).join('');

    } catch (e) {
        console.error('加载对话选项失败:', e);
    }
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message || isStreaming) return;

    input.value = '';
    autoResizeTextarea(input);

    // 移除欢迎页
    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    // 添加用户消息
    appendMessage('user', message);

    chatHistory.push({ role: 'user', content: message });

    const model = document.getElementById('playgroundModel').value;
    const provider = document.getElementById('playgroundProvider').value;
    const stream = document.getElementById('playgroundStream').checked;
    const temperature = parseFloat(document.getElementById('playgroundTemp').value);

    isStreaming = true;
    document.getElementById('sendBtn').disabled = true;

    // 创建 AI 回复气泡
    const assistantBubble = appendMessage('assistant', '');
    const bubbleContent = assistantBubble.querySelector('.chat-bubble');

    try {
        if (stream) {
            const res = await fetch(`${API_BASE}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages: chatHistory,
                    stream: true,
                    temperature,
                    provider: provider || undefined,
                }),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let assistantContent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value, { stream: true });
                const lines = text.split('\n');

                for (const line of lines) {
                    if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        const content = data.choices?.[0]?.delta?.content;
                        if (content) {
                            assistantContent += content;
                            bubbleContent.textContent = assistantContent;
                            scrollChat();
                        }
                    } catch {}
                }
            }

            if (assistantContent) {
                chatHistory.push({ role: 'assistant', content: assistantContent });
            }

        } else {
            const res = await fetch(`${API_BASE}/api/chat/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages: chatHistory,
                    stream: false,
                    temperature,
                    provider: provider || undefined,
                }),
            });

            const data = await res.json();
            const content = data.content || data.choices?.[0]?.message?.content || '(无回复)';
            bubbleContent.textContent = content;
            chatHistory.push({ role: 'assistant', content });
        }
    } catch (e) {
        bubbleContent.textContent = `错误: ${e.message}`;
        bubbleContent.style.color = 'var(--accent-red)';
    }

    isStreaming = false;
    document.getElementById('sendBtn').disabled = false;
    scrollChat();
}

function appendMessage(role, content) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    div.innerHTML = `
        <div class="chat-avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="chat-bubble">${escapeHtml(content)}</div>
    `;
    container.appendChild(div);
    scrollChat();
    return div;
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = `
        <div class="chat-welcome">
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <h3>对话测试</h3>
            <p>选择模型并开始对话，测试API流量池的响应效果</p>
        </div>
    `;
}

function scrollChat() {
    const container = document.getElementById('chatMessages');
    container.scrollTop = container.scrollHeight;
}

function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ============ API密钥 ============

let apiKeys = [];

function generateApiKey() {
    const key = 'fp-' + Array.from(crypto.getRandomValues(new Uint8Array(24)), b => b.toString(16).padStart(2, '0')).join('');
    const name = `密钥-${apiKeys.length + 1}`;
    const now = new Date().toLocaleString('zh-CN');

    apiKeys.push({ name, key, created: now, usage: 0, active: true });
    renderApiKeys();
    showToast('API密钥已生成', 'success');
}

function renderApiKeys() {
    const tbody = document.getElementById('apiKeysTableBody');
    if (apiKeys.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state"><p>暂无API密钥。当前为开放访问模式。</p></td></tr>`;
        return;
    }

    tbody.innerHTML = apiKeys.map((k, i) => `
        <tr>
            <td>${k.name}</td>
            <td>
                <code style="font-size:12px;color:var(--text-accent);font-family:'JetBrains Mono',monospace;">${k.key.slice(0, 12)}...${k.key.slice(-6)}</code>
                <button class="btn-icon" onclick="copyText('${k.key}')" title="复制">复制</button>
            </td>
            <td>${k.created}</td>
            <td>${k.usage}</td>
            <td><span class="badge ${k.active ? 'badge-success' : 'badge-error'}">${k.active ? '活跃' : '已禁用'}</span></td>
            <td>
                <button class="btn btn-sm" onclick="toggleApiKey(${i})">${k.active ? '禁用' : '启用'}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteApiKey(${i})">删除</button>
            </td>
        </tr>
    `).join('');
}

function toggleApiKey(index) {
    apiKeys[index].active = !apiKeys[index].active;
    renderApiKeys();
}

function deleteApiKey(index) {
    apiKeys.splice(index, 1);
    renderApiKeys();
    showToast('密钥已删除', 'info');
}

// ============ 工具函数 ============

function copyEndpoint() {
    copyText(`${window.location.origin}/v1`);
    showToast('API地址已复制', 'success');
}

function copyCode(id) {
    const el = document.getElementById(id);
    copyText(el.textContent);
    showToast('代码已复制', 'success');
}

function copyText(text) {
    navigator.clipboard?.writeText(text).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    });
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ============ 初始化 ============

document.addEventListener('DOMContentLoaded', () => {
    refreshDashboard();

    // textarea 自动高度
    const chatInput = document.getElementById('chatInput');
    chatInput?.addEventListener('input', function() {
        autoResizeTextarea(this);
    });

    // 每30秒自动刷新统计
    setInterval(() => {
        const activePage = document.querySelector('.page.active')?.id;
        if (activePage === 'page-dashboard') refreshDashboard();
    }, 30000);
});

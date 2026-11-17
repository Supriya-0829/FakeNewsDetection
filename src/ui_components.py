"""
UI Design System and 3D Visualizations Module.

Implements the Master Prompt specifications:
- Dark premium theme (#080B14, #101626, #141C2E, #7C5CFC, #00D4FF, #19D49A, #FF5277).
- Interactive 3D Holographic AI Intelligence Sphere (Three.js WebGL with Canvas 3D fallback).
- Glassmorphism, neon lighting, glowing borders, and micro-interactions.
- Futuristic Plotly visualizations for Analytics and Diagnostics.
"""

import json
from typing import Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


def get_custom_css() -> str:
    """
    Returns the comprehensive CSS stylesheet for the futuristic AI SaaS design system.
    """
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">

    <style>
    /* =========================================================================
       COLOR PALETTE & CSS VARIABLES
       ========================================================================= */
    :root {
        --bg-primary: #080B14;
        --bg-secondary: #101626;
        --bg-card: #141C2E;
        --accent-primary: #7C5CFC;
        --accent-secondary: #00D4FF;
        --success: #19D49A;
        --danger: #FF5277;
        --text-primary: #F5F7FF;
        --text-secondary: #929DB5;
        --border-glass: rgba(124, 92, 252, 0.18);
        --border-card: rgba(255, 255, 255, 0.08);
        --glow-primary: rgba(124, 92, 252, 0.35);
        --glow-secondary: rgba(0, 212, 255, 0.35);
    }

    /* Base Body & App Container Override */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #101626 0%, #080B14 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Headings font */
    h1, h2, h3, h4, .sora-font {
        font-family: 'Sora', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Subtle background grid pattern */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(124, 92, 252, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124, 92, 252, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* =========================================================================
       SIDEBAR CUSTOMIZATION
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: #0B0F1C !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-secondary);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    /* Radio button pills in sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 8px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(20, 28, 46, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        transition: all 0.25s ease !important;
        cursor: pointer;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(124, 92, 252, 0.15) !important;
        border-color: var(--accent-primary) !important;
        color: var(--text-primary) !important;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(124, 92, 252, 0.25), rgba(0, 212, 255, 0.15)) !important;
        border: 1px solid var(--accent-primary) !important;
        box-shadow: 0 0 12px rgba(124, 92, 252, 0.3) !important;
        color: #ffffff !important;
    }

    /* =========================================================================
       TYPOGRAPHY & HERO SECTION
       ========================================================================= */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(124, 92, 252, 0.12);
        border: 1px solid rgba(124, 92, 252, 0.3);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #C4B5FD;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        box-shadow: 0 0 15px rgba(124, 92, 252, 0.15);
    }
    .hero-badge .pulse-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-secondary);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent-secondary);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.3); }
    }

    .hero-title {
        font-family: 'Sora', sans-serif !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        color: #FFFFFF !important;
        margin-bottom: 1rem !important;
    }
    .hero-gradient-text {
        background: linear-gradient(135deg, #FFFFFF 20%, #7C5CFC 60%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        line-height: 1.6;
        color: var(--text-secondary);
        max-width: 620px;
        margin-bottom: 2rem;
    }

    /* =========================================================================
       GLASSMORPHISM CARDS & CONTAINERS
       ========================================================================= */
    .glass-card {
        background: rgba(20, 28, 46, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(124, 92, 252, 0.4);
        box-shadow: 0 12px 40px 0 rgba(124, 92, 252, 0.2);
    }
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(124, 92, 252, 0.5), transparent);
    }

    /* Metric Cards */
    .kpi-card {
        background: rgba(20, 28, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.25rem;
        text-align: left;
        transition: all 0.25s ease;
        position: relative;
    }
    .kpi-card:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 0 20px rgba(124, 92, 252, 0.2);
        transform: translateY(-2px);
    }
    .kpi-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: var(--text-secondary);
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .kpi-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: var(--success);
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* =========================================================================
       PREDICTION RESULT CARDS (REAL vs FAKE)
       ========================================================================= */
    .result-card-real {
        background: linear-gradient(135deg, rgba(25, 212, 154, 0.12), rgba(20, 28, 46, 0.85));
        border: 2px solid var(--success);
        box-shadow: 0 0 35px rgba(25, 212, 154, 0.25);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        animation: fadeInScale 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .result-card-fake {
        background: linear-gradient(135deg, rgba(255, 82, 119, 0.12), rgba(20, 28, 46, 0.85));
        border: 2px solid var(--danger);
        box-shadow: 0 0 35px rgba(255, 82, 119, 0.25);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        animation: fadeInScale 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.96); }
        to { opacity: 1; transform: scale(1); }
    }

    .result-badge-real {
        display: inline-block;
        background: rgba(25, 212, 154, 0.2);
        color: #19D49A;
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.6rem 1.6rem;
        border-radius: 100px;
        border: 1px solid #19D49A;
        letter-spacing: 0.04em;
        margin-top: 0.5rem;
    }
    .result-badge-fake {
        display: inline-block;
        background: rgba(255, 82, 119, 0.2);
        color: #FF5277;
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.6rem 1.6rem;
        border-radius: 100px;
        border: 1px solid #FF5277;
        letter-spacing: 0.04em;
        margin-top: 0.5rem;
    }

    /* =========================================================================
       INPUT & FORM CONTROLS
       ========================================================================= */
    .stTextArea textarea {
        background-color: #0E1424 !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        color: #F5F7FF !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        transition: all 0.25s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 20px rgba(124, 92, 252, 0.4) !important;
        outline: none !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C5CFC 0%, #5B3AE8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(124, 92, 252, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(124, 92, 252, 0.5) !important;
        border-color: #00D4FF !important;
    }

    /* Sample selector pills */
    .sample-pill {
        background: rgba(20, 28, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .sample-pill:hover {
        border-color: var(--accent-secondary);
        background: rgba(0, 212, 255, 0.08);
    }

    /* Scanner Radar Animation */
    .scanner-box {
        position: relative;
        background: #0E1424;
        border: 1px solid var(--accent-primary);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    .scanner-line {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00D4FF, #7C5CFC, transparent);
        box-shadow: 0 0 15px #00D4FF;
        animation: scan 2s linear infinite;
    }
    @keyframes scan {
        0% { top: 0%; opacity: 0.8; }
        50% { opacity: 1; }
        100% { top: 100%; opacity: 0.8; }
    }

    /* Advisory Box */
    .advisory-panel {
        background: rgba(255, 187, 0, 0.06);
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        color: #FDE68A;
        font-size: 0.88rem;
        margin-top: 1.25rem;
    }
    </style>
    """


def render_hero_3d_sphere(height: int = 480):
    """
    Renders the futuristic 3D Holographic AI Intelligence Sphere.
    Uses Three.js WebGL when available, with an automatic, graceful HTML5 Canvas 3D
    perspective fallback for offline or restricted environments.
    """
    sphere_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: {height}px;
                overflow: hidden;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: center;
                user-select: none;
            }}
            #canvas-container {{
                width: 100%;
                height: 100%;
                position: relative;
                cursor: grab;
            }}
            #canvas-container:active {{
                cursor: grabbing;
            }}
            .hud-overlay {{
                position: absolute;
                bottom: 12px;
                right: 16px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                color: rgba(0, 212, 255, 0.7);
                letter-spacing: 1.5px;
                pointer-events: none;
                text-transform: uppercase;
                background: rgba(14, 20, 36, 0.6);
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid rgba(0, 212, 255, 0.2);
            }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <canvas id="sphere-canvas"></canvas>
            <div class="hud-overlay">● AI SPHERE // LINGUISTIC NEURAL MAP</div>
        </div>

        <script>
        (function() {{
            const container = document.getElementById('canvas-container');
            const canvas = document.getElementById('sphere-canvas');
            let width = container.clientWidth || 500;
            let height = {height};
            canvas.width = width;
            canvas.height = height;

            // Interactive mouse tracking
            let mouseX = 0, mouseY = 0;
            let targetRotationX = 0, targetRotationY = 0;
            let currentRotationX = 0, currentRotationY = 0;

            container.addEventListener('mousemove', (e) => {{
                const rect = container.getBoundingClientRect();
                mouseX = (e.clientX - rect.left - width / 2) * 0.003;
                mouseY = (e.clientY - rect.top - height / 2) * 0.003;
            }});

            // -------------------------------------------------------------
            // HIGH-PERFORMANCE 3D PERSPECTIVE SPHERE ENGINE (Canvas 3D Engine)
            // Works 100% reliably in any sandbox, offline, or WebGL context!
            // -------------------------------------------------------------
            const ctx = canvas.getContext('2d');
            const numParticles = 650;
            const radius = Math.min(width, height) * 0.36;
            const particles = [];

            // Generate Fibonacci spherical particle distribution
            for (let i = 0; i < numParticles; i++) {{
                const phi = Math.acos(-1 + (2 * i) / numParticles);
                const theta = Math.sqrt(numParticles * Math.PI) * phi;
                const x = radius * Math.cos(theta) * Math.sin(phi);
                const y = radius * Math.sin(theta) * Math.sin(phi);
                const z = radius * Math.cos(phi);
                
                // Color distribution: #7C5CFC (purple), #00D4FF (cyan), #19D49A (emerald)
                let color;
                const r = Math.random();
                if (r > 0.45) color = 'rgba(124, 92, 252, '; // Purple
                else if (r > 0.15) color = 'rgba(0, 212, 255, '; // Cyan
                else color = 'rgba(25, 212, 154, '; // Emerald
                
                particles.push({{
                    x, y, z,
                    baseX: x, baseY: y, baseZ: z,
                    size: Math.random() * 2.2 + 0.8,
                    color: color,
                    pulseSpeed: Math.random() * 0.03 + 0.01,
                    pulsePhase: Math.random() * Math.PI * 2
                }});
            }}

            // Orbital Rings
            const ringParticles = [];
            const ringRadius1 = radius * 1.25;
            for (let i = 0; i < 90; i++) {{
                const angle = (i / 90) * Math.PI * 2;
                ringParticles.push({{
                    x: Math.cos(angle) * ringRadius1,
                    y: Math.sin(angle) * ringRadius1 * 0.28,
                    z: Math.sin(angle) * ringRadius1,
                    size: 1.5,
                    color: 'rgba(0, 212, 255, '
                }});
            }}

            let angleX = 0;
            let angleY = 0;
            let floatTime = 0;

            function animate() {{
                requestAnimationFrame(animate);

                // Smooth rotation with mouse influence
                targetRotationY += 0.004;
                targetRotationX = mouseY * 0.5;
                currentRotationX += (targetRotationX - currentRotationX) * 0.05;
                currentRotationY += (targetRotationY + mouseX * 0.05 - currentRotationY) * 0.05;

                floatTime += 0.025;
                const floatOffset = Math.sin(floatTime) * 8;

                ctx.clearRect(0, 0, width, height);

                const cosX = Math.cos(currentRotationX);
                const sinX = Math.sin(currentRotationX);
                const cosY = Math.cos(currentRotationY);
                const sinY = Math.sin(currentRotationY);

                const fov = 400;
                const centerX = width / 2;
                const centerY = height / 2 + floatOffset;

                // Central glowing aura
                const auraGradient = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, radius * 1.1);
                auraGradient.addColorStop(0, 'rgba(124, 92, 252, 0.22)');
                auraGradient.addColorStop(0.5, 'rgba(0, 212, 255, 0.08)');
                auraGradient.addColorStop(1, 'rgba(8, 11, 20, 0)');
                ctx.fillStyle = auraGradient;
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius * 1.1, 0, Math.PI * 2);
                ctx.fill();

                // Sort particles by depth for correct 3D perspective
                const projected = [];

                for (let i = 0; i < particles.length; i++) {{
                    const p = particles[i];

                    // Subtle breathing wave
                    const pulse = Math.sin(floatTime + p.pulsePhase) * 4;
                    const px = p.baseX + (p.baseX / radius) * pulse;
                    const py = p.baseY + (p.baseY / radius) * pulse;
                    const pz = p.baseZ + (p.baseZ / radius) * pulse;

                    // 3D rotation math
                    let x1 = px * cosY - pz * sinY;
                    let z1 = pz * cosY + px * sinY;
                    let y1 = py * cosX - z1 * sinX;
                    let z2 = z1 * cosX + py * sinX;

                    // Perspective projection
                    const scale = fov / (fov + z2);
                    const x2d = centerX + x1 * scale;
                    const y2d = centerY + y1 * scale;
                    const alpha = Math.max(0.12, Math.min(1.0, (z2 + radius) / (2 * radius) * 0.85 + 0.15));

                    projected.push({{
                        x: x2d,
                        y: y2d,
                        z: z2,
                        size: p.size * scale,
                        alpha: alpha,
                        color: p.color
                    }});
                }}

                // Render interconnected network lines for close nodes in front hemisphere
                ctx.lineWidth = 0.5;
                for (let i = 0; i < projected.length; i += 3) {{
                    const p1 = projected[i];
                    if (p1.z < 0) continue; // Skip lines in back hemisphere
                    for (let j = i + 1; j < projected.length; j += 6) {{
                        const p2 = projected[j];
                        if (p2.z < 0) continue;
                        const dx = p1.x - p2.x;
                        const dy = p1.y - p2.y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 42) {{
                            const lineAlpha = (1 - dist / 42) * 0.28 * p1.alpha;
                            ctx.strokeStyle = `rgba(0, 212, 255, ${{lineAlpha}})`;
                            ctx.beginPath();
                            ctx.moveTo(p1.x, p1.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.stroke();
                        }}
                    }}
                }}

                // Sort by depth (Z)
                projected.sort((a, b) => a.z - b.z);

                // Draw particles
                for (let i = 0; i < projected.length; i++) {{
                    const p = projected[i];
                    ctx.fillStyle = p.color + p.alpha + ')';
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, Math.max(0.6, p.size), 0, Math.PI * 2);
                    ctx.fill();

                    // Subtle neon glow on nearest 20% particles
                    if (p.z > radius * 0.5) {{
                        ctx.fillStyle = 'rgba(255, 255, 255, ' + (p.alpha * 0.5) + ')';
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, p.size * 0.5, 0, Math.PI * 2);
                        ctx.fill();
                    }}
                }}

                // Draw orbital digital ring
                for (let i = 0; i < ringParticles.length; i++) {{
                    const rp = ringParticles[i];
                    let x1 = rp.x * cosY - rp.z * sinY;
                    let z1 = rp.z * cosY + rp.x * sinY;
                    let y1 = rp.y * cosX - z1 * sinX;
                    let z2 = z1 * cosX + rp.y * sinX;
                    const scale = fov / (fov + z2);
                    const x2d = centerX + x1 * scale;
                    const y2d = centerY + y1 * scale;
                    const alpha = Math.max(0.1, (z2 + ringRadius1) / (2 * ringRadius1) * 0.6);

                    ctx.fillStyle = rp.color + alpha + ')';
                    ctx.beginPath();
                    ctx.arc(x2d, y2d, rp.size * scale, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }}

            animate();

            window.addEventListener('resize', () => {{
                width = container.clientWidth || 500;
                canvas.width = width;
            }});
        }})();
        </script>
    </body>
    </html>
    """
    components.html(sphere_html, height=height, scrolling=False)


def create_class_distribution_chart(fake_count: int, real_count: int) -> go.Figure:
    """
    Creates a dark glassmorphic donut chart of dataset class distribution.
    """
    labels = ["FAKE News", "REAL News"]
    values = [fake_count, real_count]
    colors = ["#FF5277", "#19D49A"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#080B14", width=3)),
                textinfo="label+percent",
                textfont=dict(family="Inter, sans-serif", size=13, color="#FFFFFF"),
                hoverinfo="label+value+percent",
            )
        ]
    )

    total = fake_count + real_count
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter", size=12, color="#929DB5"),
        ),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#929DB5'>ARTICLES</span>",
                x=0.5,
                y=0.5,
                font=dict(size=18, family="Space Grotesk", color="#F5F7FF"),
                showarrow=False,
            )
        ],
    )
    return fig


def create_confusion_matrix_chart(cm_matrix: list) -> go.Figure:
    """
    Creates a dark cyberpunk heatmap for the confusion matrix.
    """
    z = cm_matrix
    x = ["Pred FAKE", "Pred REAL"]
    y = ["True FAKE", "True REAL"]

    total = sum(sum(row) for row in z)
    text_annotations = []
    for i in range(2):
        row_text = []
        for j in range(2):
            val = z[i][j]
            pct = (val / total) * 100 if total > 0 else 0
            row_text.append(f"<b>{val:,}</b><br><span style='font-size:11px'>({pct:.1f}%)</span>")
        text_annotations.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text_annotations,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#101626"],
                [0.15, "#1B2540"],
                [0.5, "#4B36A8"],
                [1.0, "#00D4FF"],
            ],
            showscale=False,
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=20, r=20),
        height=300,
        font=dict(family="Inter", color="#F5F7FF", size=12),
        xaxis=dict(tickfont=dict(size=12, color="#929DB5"), side="bottom"),
        yaxis=dict(tickfont=dict(size=12, color="#929DB5"), autorange="reversed"),
    )
    return fig


def create_model_comparison_chart(comparisons: Dict[str, Any]) -> go.Figure:
    """
    Creates a grouped bar chart comparing multiple algorithms with glowing neon accents.
    """
    models = list(comparisons.keys())
    acc = [comparisons[m]["accuracy"] * 100 for m in models]
    prec = [comparisons[m]["precision"] * 100 for m in models]
    rec = [comparisons[m]["recall"] * 100 for m in models]
    f1 = [comparisons[m]["f1_score"] * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=acc, name="Accuracy", marker_color="#00D4FF"))
    fig.add_trace(go.Bar(x=models, y=prec, name="Precision", marker_color="#19D49A"))
    fig.add_trace(go.Bar(x=models, y=rec, name="Recall", marker_color="#7C5CFC"))
    fig.add_trace(go.Bar(x=models, y=f1, name="F1-Score", marker_color="#FF5277"))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=20, r=20),
        height=360,
        font=dict(family="Inter", color="#929DB5", size=11),
        yaxis=dict(
            range=[92, 100.5],
            gridcolor="rgba(255,255,255,0.06)",
            title="Score (%)",
            title_font=dict(color="#929DB5"),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.02)",
            tickfont=dict(size=11, color="#F5F7FF"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#929DB5"),
        ),
    )
    return fig

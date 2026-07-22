console.log("🎨 Monet starting");

const Monet = {
    socket: null,
    borders: new Map(),       // id -> border element (pending results)
    videoResults: new Map(),  // video element -> { src, score, reason, borderId }
    lastUrl: location.href,   // Track URL for SPA navigation
    reconnectTimer: null,

    init() {
        this.addStyles();
        this.connect();
        this.watchVideos();
        this.watchStaleBorders();
    },

    connect() {
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        try {
            this.socket = new WebSocket('ws://localhost:8000/ws');
        } catch (e) {
            console.log("🎨 Server not running");
            this.scheduleReconnect();
            return;
        }

        this.socket.onopen = () => {
            console.log("🎨 Connected to server");
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
        };

        this.socket.onclose = () => {
            console.log("🎨 Disconnected");
            this.scheduleReconnect();
        };

        this.socket.onerror = () => {
            // onclose will fire after this, so reconnect is handled there
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (!data.videoId || data.ai_score === undefined) {
                    console.warn("🎨 Invalid response:", data);
                    return;
                }

                let border = this.borders.get(data.videoId);

                // If border was removed from DOM, try to re-create it
                if (!border || !border.isConnected) {
                    for (const [video, info] of this.videoResults.entries()) {
                        if (info.borderId === data.videoId) {
                            const container = this.getContainer(video);
                            if (container) {
                                // Remove any stale border first
                                const existing = container.querySelector('div[id^="monet-"]');
                                if (existing) existing.remove();

                                border = this._createBorderElement(data.videoId, container);
                                this.borders.set(data.videoId, border);
                            }
                            break;
                        }
                    }
                }

                if (border && border.isConnected) {
                    this.updateColor(border, data);

                    // Store result so border persists across replays
                    for (const [video, info] of this.videoResults.entries()) {
                        if (info.borderId === data.videoId) {
                            info.score = data.ai_score;
                            info.label = data.label;
                            info.reason = data.detection_reason;
                            info.confidence = data.confidence_level;
                            info.data = data;
                            info.analyzed = true;
                            break;
                        }
                    }

                    border.dataset.resolved = 'true';
                } else {
                    console.warn("🎨 Border not found for", data.videoId);
                }
            } catch (e) {
                console.error("🎨 Failed to process result:", e, event.data?.substring?.(0, 200));
            }
        };
    },

    scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, 3000);
    },

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
      @keyframes monet-glow-purple {
        0%, 100% { box-shadow: 0 0 20px rgba(168, 85, 247, 0.6), 0 0 40px rgba(168, 85, 247, 0.4); }
        50% { box-shadow: 0 0 30px rgba(168, 85, 247, 0.8), 0 0 60px rgba(168, 85, 247, 0.6); }
      }
      @keyframes monet-glow-red {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.6), 0 0 40px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.9), 0 0 80px rgba(239, 68, 68, 0.7); }
      }
      @keyframes monet-glow-orange {
        0%, 100% { box-shadow: 0 0 20px rgba(249, 115, 22, 0.6), 0 0 40px rgba(249, 115, 22, 0.4); }
        50% { box-shadow: 0 0 35px rgba(249, 115, 22, 0.8), 0 0 70px rgba(249, 115, 22, 0.6); }
      }
      @keyframes monet-glow-green {
        0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.6), 0 0 40px rgba(34, 197, 94, 0.4); }
        50% { box-shadow: 0 0 35px rgba(34, 197, 94, 0.8), 0 0 70px rgba(34, 197, 94, 0.6); }
      }
      @keyframes monet-spin { 100% { transform: rotate(360deg); } }
      .monet-spin-icon { animation: monet-spin 2s linear infinite; }
      
      .monet-info-wrapper { display: flex; align-items: center; justify-content: center; pointer-events: none; }
      .monet-info-icon { opacity: 1; cursor: default; transition: opacity 0.2s; filter: drop-shadow(0 1px 3px rgba(0,0,0,0.4)); }
      .monet-hover-pill {
          position: absolute;
          top: 4px; left: 4px; right: 4px; bottom: 4px;
          border-radius: 72px;
          transition: background-color 0.2s;
          pointer-events: auto;
          cursor: pointer;
      }
      .monet-hover-pill:hover {
          background-color: rgba(255, 255, 255, 0.15);
      }
      .monet-hover-pill.monet-popup-open {
          background-color: rgba(255, 255, 255, 0.18);
      }
      .monet-hover-pill.monet-popup-open ~ .monet-breakdown-popup { opacity: 1; transform: translate(-50%, 0); pointer-events: auto; visibility: visible; }
      
      .monet-breakdown-popup {
          position: absolute;
          top: 100%;
          left: 50%;
          margin-top: 12px;
          width: 250px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 24px;
          padding: 18px;
          color: white;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          text-shadow: 0 1px 3px rgba(0,0,0,0.4);
          opacity: 0;
          transform: translate(-50%, -8px);
          visibility: hidden;
          pointer-events: none;
          transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          z-index: 999;
          cursor: default;
      }
      .monet-breakdown-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.05); }
      .monet-breakdown-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
      .monet-breakdown-title { font-weight: 600; color: white; letter-spacing: 0.2px; }
      .monet-breakdown-val { font-weight: 600; font-family: monospace; font-size: 13.5px; }
      .monet-breakdown-header { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: white; margin-bottom: 10px; font-weight: 700; }
    `;
        document.head.appendChild(style);
    },

    // Check if a video needs analysis and trigger it
    tryAnalyze(video) {
        const result = this.videoResults.get(video);
        // Already analyzed — keep border
        if (result && result.analyzed) return;
        // Currently analyzing — don't double-send
        if (result && result.borderId && !result.analyzed) return;
        // Analyze after brief delay for frames to stabilize
        setTimeout(() => this.analyze(video), 300);
    },

    watchVideos() {
        setInterval(() => {
            // Detect SPA navigation (URL changed but page didn't reload)
            if (location.href !== this.lastUrl) {
                this.lastUrl = location.href;
            }

            document.querySelectorAll('video').forEach(video => {
                const currentSrc = video.currentSrc || video.src;

                // Check if this video's source changed (new video loaded in same element)
                const cached = this.videoResults.get(video);
                if (cached && cached.src && currentSrc && cached.src !== currentSrc) {
                    this.resetVideo(video);
                    // Re-analyze after scroll animation settles
                    setTimeout(() => this.tryAnalyze(video), 800);
                }

                if (video.monetChecked) return;
                video.monetChecked = true;

                // Attach play listener for future plays
                video.addEventListener('play', () => this.tryAnalyze(video));

                // If video is ALREADY playing, analyze it now
                if (!video.paused && !video.ended && video.readyState >= 2) {
                    this.tryAnalyze(video);
                }
            });
        }, 1000);
    },

    // Clean up borders that have been "analyzing" for too long (>30s = server never replied)
    watchStaleBorders() {
        setInterval(() => {
            for (const [id, border] of this.borders.entries()) {
                if (!border.isConnected) {
                    this.borders.delete(id);
                    continue;
                }
                // If border has been unresolved for >30s, remove it and allow re-analysis
                if (!border.dataset.resolved && border.dataset.created) {
                    const age = Date.now() - parseInt(border.dataset.created);
                    if (age > 30000) {
                        console.log("🎨 Removing stale border:", id);
                        border.remove();
                        this.borders.delete(id);
                        // Allow the video to be re-analyzed
                        for (const [video, info] of this.videoResults.entries()) {
                            if (info.borderId === id) {
                                this.videoResults.delete(video);
                                video.monetChecked = false;
                                break;
                            }
                        }
                    }
                }
            }
        }, 10000);
    },

    // Reset everything (used on URL change)
    resetAll() {
        document.querySelectorAll('video').forEach(video => {
            this.resetVideo(video);
            video.monetChecked = false;
        });
        this.borders.clear();
        this.videoResults.clear();
    },

    // Reset a single video's overlay and cached result
    resetVideo(video) {
        const container = this.getContainer(video);
        if (!container) return;

        const existingBorder = container.querySelector('div[id^="monet-"]');
        if (existingBorder) {
            this.borders.delete(existingBorder.id);
            existingBorder.remove();
        }

        const existingLabel = container.querySelector('.monet-label');
        if (existingLabel) existingLabel.remove();

        this.videoResults.delete(video);
        video.monetChecked = false;
    },

    getContainer(video) {
        return video.closest('[class*="reel"]')
            || video.closest('[class*="shorts"]')
            || video.closest('article')
            || video.parentElement?.parentElement
            || video.parentElement;
    },

    collectMetadata() {
        const metaTitle = document.querySelector('meta[name="title"]')?.content
            || document.querySelector('meta[property="og:title"]')?.content
            || '';
        const title = document.querySelector('h1 yt-formatted-string')?.textContent
            || document.querySelector('h1')?.textContent
            || metaTitle;
        const channel = document.querySelector('#channel-name a')?.textContent
            || document.querySelector('ytd-channel-name a')?.textContent
            || '';
        const description = document.querySelector('meta[name="description"]')?.content
            || document.querySelector('meta[property="og:description"]')?.content
            || '';
        const hashtags = Array.from(document.querySelectorAll('a[href*="hashtag"], a[href*="/hashtag/"]'))
            .map(a => a.textContent.trim())
            .filter(Boolean)
            .slice(0, 12);
        const pageText = (document.body?.innerText || '')
            .split('\n')
            .map(s => s.trim())
            .filter(Boolean)
            .slice(0, 80)
            .join(' ')
            .slice(0, 3000);
        const youtubeAiDisclosure = this.collectYoutubeAiDisclosure();

        return {
            title: (title || '').trim(),
            channel: (channel || '').trim(),
            description: (description || '').trim(),
            hashtags,
            pageText,
            youtubeAiDisclosure,
            url: location.href
        };
    },

    collectYoutubeAiDisclosure() {
        const candidates = Array.from(document.querySelectorAll(
            '[aria-label], [title], button, yt-formatted-string, span, div'
        ));
        // Only treat explicit disclosure phrases as evidence. Previously a
        // bare "AI" / "synthetic" / "altered" exact match (isPillText) fired
        // on random UI elements (menus, tooltips, suggestions, "how this
        // content was made" links), flagging EVERY video as 100% AI without
        // ever sending frames to the server.
        const DISCLOSURE_PHRASES = [
            'altered or synthetic content',
            'altered content',
            'synthetic content',
            'synthetic media',
            'ai-generated',
            'ai generated',
            'created with ai',
            'made with ai',
            'generated with ai'
        ];

        for (const element of candidates) {
            // Exclude video titles, channel names, descriptions, and comments to avoid false positives
            if (
                element.closest('h1') ||
                element.closest('#title') ||
                element.closest('#video-title') ||
                element.closest('[class*="title"]') ||
                element.closest('#channel-name') ||
                element.closest('ytd-channel-name') ||
                element.closest('#description') ||
                element.closest('ytd-comment-renderer') ||
                element.closest('ytd-comment-thread-renderer')
            ) {
                continue;
            }

            const text = [
                element.getAttribute('aria-label') || '',
                element.getAttribute('title') || '',
                element.textContent || ''
            ].join(' ').replace(/\s+/g, ' ').trim();

            // Disclosure copy is short — skip long blobs (descriptions, transcripts).
            if (!text || text.length > 120) continue;

            const lower = text.toLowerCase();

            if (DISCLOSURE_PHRASES.some(phrase => lower.includes(phrase))) {
                return text;
            }
        }

        return '';
    },

    _createBorderElement(id, container) {
        if (getComputedStyle(container).position === 'static') {
            container.style.position = 'relative';
        }

        const border = document.createElement('div');
        border.id = id;
        border.dataset.created = Date.now().toString();
        border.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      border-radius: 12px;
      pointer-events: none;
      z-index: 2147483646;
      border: 4px solid rgba(168, 85, 247, 0.5);
      opacity: 0.5;
      transition: all 0.5s ease;
    `;

        container.appendChild(border);

        // Add "Scanning..." floating label immediately
        let label = container.querySelector('.monet-label');
        if (!label) {
            label = document.createElement('div');
            label.className = 'monet-label';
            label.style.cssText = `
                position: absolute;
                top: 16px;
                left: 46%;
                height: 48px;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 0 12px;
                border-radius: 72px;
                font-family: "Roboto", "Arial", sans-serif;
                color: white;
                z-index: 2147483647;
                pointer-events: auto;
                transition: opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                transform: translateX(-50%) translateY(0);
                background: rgba(0, 0, 0, 0.3);
            `;

            const scanSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="rgba(168, 85, 247, 1.0)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="monet-info-icon monet-spin-icon"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>`;

            label.innerHTML = `
                <div class="monet-hover-pill"></div>
                <div class="monet-info-wrapper" style="position:relative; z-index:1; opacity: 0.5;">
                    ${scanSvg}
                </div>
                <div style="position:relative; z-index:1; opacity: 0.5; font-weight: 700; font-size: 16px; letter-spacing: 0.4px; line-height: 1; text-shadow: 0 1px 3px rgba(0,0,0,0.4);">
                    Scanning...
                </div>
            `;
            container.appendChild(label);
        }

        return border;
    },

    async analyze(video) {
        const rect = video.getBoundingClientRect();
        if (rect.height < rect.width) return;

        // Skip videos that aren't visible in the viewport (YouTube pre-loads next/prev Shorts)
        const visibleHeight = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
        if (visibleHeight < rect.height * 0.5) return;

        const container = this.getContainer(video);
        if (!container) return;

        // Don't create if already has a border
        if (container.querySelector('div[id^="monet-"]')) return;

        // Check WebSocket is connected
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.connect();
            return;
        }

        const id = 'monet-' + Date.now();

        const border = this._createBorderElement(id, container);
        this.borders.set(id, border);

        const metadata = this.collectMetadata();

        if (metadata.youtubeAiDisclosure) {
            this.updateColor(border, {
                ai_score: 1.0,
                confidence_level: 'High',
                detection_reason: 'YouTube AI disclosure detected: ' + metadata.youtubeAiDisclosure,
                label: 'STRONG AI EVIDENCE',
                breakdown: {
                    'metadata': { score: 1.0 }
                }
            });
            this.videoResults.set(video, {
                src: video.currentSrc || video.src,
                borderId: id,
                score: 1.0,
                reason: 'YouTube AI disclosure detected: ' + metadata.youtubeAiDisclosure,
                metadata,
                analyzed: true,
                data: {
                    ai_score: 1.0,
                    confidence_level: 'High',
                    detection_reason: 'YouTube AI disclosure detected: ' + metadata.youtubeAiDisclosure,
                    label: 'STRONG AI EVIDENCE',
                    breakdown: {
                        'metadata': { score: 1.0 }
                    }
                }
            });
            return;
        }

        // Cache video info for persistence
        this.videoResults.set(video, {
            src: video.currentSrc || video.src,
            borderId: id,
            score: null,
            reason: null,
            metadata,
            analyzed: false
        });

        // Capture a wider slice of the Short so temporal artifacts have a chance to show up.
        const frames = [];
        const frameCount = 8;
        const frameDelayMs = 600;
        for (let i = 0; i < frameCount; i++) {
            if (i > 0) await new Promise(r => setTimeout(r, frameDelayMs));
            if (!video.isConnected) break;
            const c = document.createElement('canvas');
            c.width = 480;
            c.height = 854;
            c.getContext('2d').drawImage(video, 0, 0, 480, 854);
            frames.push(c.toDataURL('image/jpeg', 0.85));
        }

        if (frames.length < 4) {
            border.remove();
            this.borders.delete(id);
            this.videoResults.delete(video);
            video.monetChecked = false;
            return;
        }

        try {
            this.socket.send(JSON.stringify({
                type: "analyze",
                frames: frames,
                videoId: id,
                metadata
            }));
        } catch (e) {
            console.error("🎨 Send failed:", e);
            border.remove();
            this.borders.delete(id);
            this.videoResults.delete(video);
            video.monetChecked = false;
        }
    },

    updateColor(border, data) {
        const score = data.ai_score;
        const confidence = data.confidence_level || 'Medium';
        let color, glowAnim, labelText, labelBg;

        // High AI evidence
        if (score >= 0.58) {
            color = 'rgba(239, 68, 68, 1.0)';
            glowAnim = 'monet-glow-red';
            labelText = 'Likely AI';
            labelBg = 'rgba(220, 38, 38, 0.85)';
        }
        // Mixed AI evidence
        else if (score >= 0.32) {
            color = 'rgba(249, 115, 22, 1.0)';
            glowAnim = 'monet-glow-orange';
            labelText = 'Suspicious';
            labelBg = 'rgba(234, 88, 12, 0.85)';
        }
        // Low AI evidence
        else {
            color = 'rgba(34, 197, 94, 0.9)';
            glowAnim = 'monet-glow-green';
            labelText = 'Low AI';
            labelBg = 'rgba(22, 163, 74, 0.85)';
        }

        // Update border
        border.style.borderColor = labelBg;
        const glowColor = color.replace('1.0)', '0.6)').replace('0.9)', '0.6)');
        border.style.boxShadow = `0 0 20px ${glowColor}, 0 0 40px ${glowColor.replace('0.6)', '0.4)')}, inset 0 0 20px ${glowColor.replace('0.6)', '0.2)')}`;
        border.style.opacity = '1';
        border.style.animation = `${glowAnim} 2s ease-in-out infinite`;

        // Add floating label
        let label = border.parentElement.querySelector('.monet-label');
        if (!label) {
            label = document.createElement('div');
            label.className = 'monet-label';
            label.style.cssText = `
                position: absolute;
                top: 16px;
                left: 46%;
                height: 48px;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 0 12px;
                border-radius: 72px;
                font-family: "Roboto", "Arial", sans-serif;
                color: white;
                z-index: 2147483647;
                pointer-events: auto;
                transition: opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                transform: translateX(-50%) translateY(0);
            `;
            border.parentElement.appendChild(label);
        }

        // Build Breakdown Popup HTML
        let breakdownHtml = `<div class="monet-breakdown-header">Analysis Breakdown</div>`;

        const getScoreColor = (val) => {
            if (val >= 0.6) return '#ef4444'; // Red
            if (val >= 0.35) return '#f97316'; // Orange
            return '#22c55e'; // Green
        };

        const formatKeyName = (key) => {
            const dictionary = {
                'texture_smoothness': 'Texture Smoothness',
                'color': 'Color Variance',
                'semantic': 'Semantics',
                'ai_detector': 'AI Detector',
                'metadata': 'Metadata / YouTube AI Label',
                'digital_penalty': 'Digital Media Penalty'
            };
            if (dictionary[key]) return dictionary[key];
            return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        };

        if (data.breakdown) {
            for (const [key, bData] of Object.entries(data.breakdown)) {
                const colorHex = getScoreColor(bData.score);
                breakdownHtml += `
                    <div class="monet-breakdown-row">
                        <span class="monet-breakdown-title">${formatKeyName(key)}</span>
                        <span class="monet-breakdown-val" style="color: ${colorHex}">${Math.round(bData.score * 100)}%</span>
                    </div>
                `;
            }
        }

        const getConfidenceColor = (conf) => {
            if (conf === 'High') return '#22c55e'; // Green
            if (conf === 'Medium') return '#f97316'; // Orange
            if (conf === 'Low') return '#ef4444'; // Red
            return '#22c55e';
        };

        const confValColor = getConfidenceColor(confidence);
        breakdownHtml += `
            <div class="monet-breakdown-row" style="margin-top: 8px;">
                <span class="monet-breakdown-title">Confidence</span>
                <span class="monet-breakdown-val" style="font-family: inherit; color: ${confValColor}">${confidence}</span>
            </div>
        `;

        const infoSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="monet-info-icon"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>`;

        label.innerHTML = `
            <div class="monet-hover-pill"></div>
            <div class="monet-info-wrapper" style="position:relative; z-index:1;">
                ${infoSvg}
            </div>
            <div style="position:relative; z-index:1; font-weight: 700; font-size: 16px; letter-spacing: 0.4px; line-height: 1; text-shadow: 0 1px 3px rgba(0,0,0,0.4);">
                ${Math.round(score * 100)}% ${labelText}
            </div>
            <div class="monet-breakdown-popup">
                ${breakdownHtml}
            </div>
        `;

        label.style.background = 'rgba(0, 0, 0, 0.3)';

        // Click to toggle dropdown
        const pill = label.querySelector('.monet-hover-pill');
        if (pill) {
            pill.addEventListener('click', event => {
                event.stopPropagation();
                const isOpen = pill.classList.toggle('monet-popup-open');
                // Close any other open pills
                if (isOpen) {
                    document.querySelectorAll('.monet-hover-pill.monet-popup-open').forEach(p => {
                        if (p !== pill) p.classList.remove('monet-popup-open');
                    });
                    // Register a one-time outside-click listener to close
                    const closeHandler = (e) => {
                        if (!label.contains(e.target)) {
                            pill.classList.remove('monet-popup-open');
                            document.removeEventListener('click', closeHandler, true);
                        }
                    };
                    document.addEventListener('click', closeHandler, true);
                }
            });
        }
    }
};

Monet.init();

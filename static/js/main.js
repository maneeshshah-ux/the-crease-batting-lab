/**
 * the CREASE Batting Lab — Main JavaScript
 *
 * UX enhancements: toast notifications, drag-and-drop upload,
 * job polling with progress, onboarding wizard, PWA install.
 */

(function() {
    'use strict';

    // ----------------------------------------------------------------
    // Toast Notification System
    // ----------------------------------------------------------------
    window.showToast = function(message, type, duration) {
        type = type || 'info';
        duration = duration || 4000;
        var container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            container.id = 'toast-container';
            document.body.appendChild(container);
        }

        var icons = {
            success: 'bi-check-circle-fill',
            error: 'bi-x-circle-fill',
            warning: 'bi-exclamation-triangle-fill',
            info: 'bi-info-circle-fill',
        };

        var toast = document.createElement('div');
        toast.className = 'toast-crease ' + type;
        toast.innerHTML = '<i class="bi ' + (icons[type] || icons.info) + ' me-2"></i> ' +
                          '<span>' + message + '</span>';
        container.appendChild(toast);

        setTimeout(function() {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(function() { toast.remove(); }, 300);
        }, duration);
    };

    // Auto-show flash messages from server as toasts
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.alert-dismissible').forEach(function(alert) {
            var message = alert.querySelector('.alert-text') || alert;
            var text = message.textContent.trim();
            var type = 'info';
            if (alert.classList.contains('alert-success')) type = 'success';
            else if (alert.classList.contains('alert-danger')) type = 'error';
            else if (alert.classList.contains('alert-warning')) type = 'warning';

            if (text) {
                setTimeout(function() {
                    window.showToast(text, type, 5000);
                    try {
                        var bs = bootstrap.Alert.getOrCreateInstance(alert);
                        bs.close();
                    } catch(e) { alert.remove(); }
                }, 500);
            }
        });
    });

    // ----------------------------------------------------------------
    // PWA Install Prompt
    // ----------------------------------------------------------------
    var deferredPrompt = null;

    window.addEventListener('beforeinstallprompt', function(e) {
        e.preventDefault();
        deferredPrompt = e;
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'block';
    });

    window.addEventListener('appinstalled', function() {
        deferredPrompt = null;
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'none';
        window.showToast('the CREASE installed! Add it to your home screen for best experience.', 'success');
    });

    window.dismissPWA = function() {
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'none';
    };

    // Show PWA install hint on mobile after a delay
    setTimeout(function() {
        var isMobile = /Android|iPhone|webOS|iPad/i.test(navigator.userAgent);
        var isStandalone = window.matchMedia('(display-mode: standalone)').matches;
        if (isMobile && !isStandalone) {
            var banner = document.getElementById('pwa-install-banner');
            if (banner && banner.style.display !== 'block') banner.style.display = 'block';
        }
    }, 5000);

    // ----------------------------------------------------------------
    // Drag-and-Drop Upload Zone
    // ----------------------------------------------------------------
    function initUploadZone() {
        var zone = document.getElementById('upload-zone');
        var fileInput = document.getElementById('video');
        if (!zone || !fileInput) return;

        // Click to upload
        zone.addEventListener('click', function() { fileInput.click(); });

        // Drag events
        ['dragenter', 'dragover'].forEach(function(evt) {
            zone.addEventListener(evt, function(e) {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(function(evt) {
            zone.addEventListener(evt, function(e) {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove('dragover');
            });
        });

        zone.addEventListener('drop', function(e) {
            var files = e.dataTransfer.files;
            if (files.length > 0) {
                var file = files[0];
                if (file.type.startsWith('video/')) {
                    fileInput.files = files;
                    updateFileInfo(file);
                } else {
                    window.showToast('Please drop a video file (MP4, MOV, etc.)', 'error');
                }
            }
        });

        // File input change
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                updateFileInfo(e.target.files[0]);
            }
        });
    }

    function updateFileInfo(file) {
        var zone = document.getElementById('upload-zone');
        var info = document.getElementById('file-info');
        if (!zone) return;

        zone.classList.add('has-file');

        if (info) {
            var sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            var icon = file.type.includes('video') ? 'bi-film' : 'bi-file-earmark';
            info.innerHTML = '<i class="bi ' + icon + '" style="color: #4CAF50; font-size: 1.5rem;"></i>' +
                            '<div class="mt-2"><strong style="color: #C2C2C2;">' + file.name + '</strong>' +
                            '<br><small style="color: #888888;">' + sizeMB + ' MB</small></div>' +
                            '<button type="button" class="btn btn-sm btn-outline-crease mt-2" ' +
                            'onclick="clearFile()"><i class="bi bi-x"></i> Remove</button>';
        }
    }

    window.clearFile = function() {
        var fileInput = document.getElementById('video');
        var zone = document.getElementById('upload-zone');
        var info = document.getElementById('file-info');
        if (fileInput) fileInput.value = '';
        if (zone) zone.classList.remove('has-file');
        if (info) info.innerHTML = '';
    };

    // Open camera capture on mobile
    window.openCamera = function() {
        var fileInput = document.getElementById('video');
        if (fileInput) {
            fileInput.setAttribute('capture', 'environment');
            fileInput.setAttribute('accept', 'video/*');
            fileInput.click();
            fileInput.removeAttribute('capture');
        }
    };

    // ----------------------------------------------------------------
    // Onboarding Flow
    // ----------------------------------------------------------------
    window.startOnboarding = function() {
        var overlay = document.getElementById('onboarding-overlay');
        if (overlay) overlay.style.display = 'flex';
    };

    window.dismissOnboarding = function() {
        var overlay = document.getElementById('onboarding-overlay');
        if (overlay) overlay.style.display = 'none';
        // Remember dismissal
        try {
            localStorage.setItem('crease_onboarding_done', 'true');
        } catch(e) {}
    };

    // Auto-show onboarding for first-time users
    document.addEventListener('DOMContentLoaded', function() {
        var hasSessions = document.querySelector('.session-table-row');
        var onboardingDone = false;
        try { onboardingDone = localStorage.getItem('crease_onboarding_done') === 'true'; } catch(e) {}

        if (!hasSessions && !onboardingDone) {
            setTimeout(function() {
                var overlay = document.getElementById('onboarding-overlay');
                if (overlay) overlay.style.display = 'flex';
            }, 1000);
        }
    });

    // ----------------------------------------------------------------
    // Job Polling (analysis progress)
    // ----------------------------------------------------------------
    window.pollJob = function(jobId) {
        if (!jobId) return;

        // Persist job ID so user can leave & come back without losing progress
        try { localStorage.setItem('crease_active_job', jobId); } catch(e) {}

        var pollInterval = setInterval(function() {
            fetch('/api/job/' + jobId)
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var progressBar = document.getElementById('progress-bar');
                    var statusText = document.getElementById('status-text');
                    var phaseText = document.getElementById('phase-text');

                    if (progressBar) {
                        // Defensive: ensure progress is always a number
                        var pct = (typeof data.progress === 'number') ? data.progress : 0;
                        progressBar.style.width = pct + '%';
                        progressBar.setAttribute('aria-valuenow', pct);
                    }
                    if (statusText) {
                        statusText.textContent = data.message || 'Processing...';
                    }
                    if (phaseText && data.message) {
                        phaseText.textContent = data.message;
                    }

                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        clearActiveJob();
                        window.showToast('Analysis complete! Viewing results...', 'success');
                        setTimeout(function() {
                            window.location.href = data.redirect || '/';
                        }, 1000);
                    } else if (data.status === 'failed') {
                        clearInterval(pollInterval);
                        clearActiveJob();
                        window.showToast(data.error || 'Analysis failed. Please try again.', 'error');
                        if (statusText) {
                            statusText.textContent = '❌ ' + (data.error || 'Analysis failed');
                            statusText.style.color = '#E55000';
                        }
                    }
                })
                .catch(function(err) {
                    console.error('Poll error:', err);
                });
        }, 1500);
    };

    // Resume a previously active job if user returns to any page
    function clearActiveJob() {
        try { localStorage.removeItem('crease_active_job'); } catch(e) {}
    }
    function resumeActiveJob() {
        try {
            var jobId = localStorage.getItem('crease_active_job');
            if (jobId && !window.location.pathname.includes('/job/')) {
                fetch('/api/job/' + jobId)
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.status === 'processing' || data.status === 'queued') {
                            // Still running — redirect back to processing page
                            window.location.href = '/job/' + jobId;
                        } else if (data.status === 'completed' || data.status === 'failed') {
                            clearActiveJob();
                        }
                    })
                    .catch(function() { /* ignore — job may have expired */ });
            }
        } catch(e) {}
    }
    // Check for active job on every page load
    document.addEventListener('DOMContentLoaded', resumeActiveJob);

    // ----------------------------------------------------------------
    // Copy to clipboard helper
    // ----------------------------------------------------------------
    window.copyToClipboard = function(text, label) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(function() {
                window.showToast((label || 'Copied') + ' to clipboard!', 'success');
            }).catch(function() {
                fallbackCopy(text, label);
            });
        } else {
            fallbackCopy(text, label);
        }
    };

    function fallbackCopy(text, label) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            window.showToast((label || 'Copied') + ' to clipboard!', 'success');
        } catch(e) {
            window.showToast('Could not copy. Please copy manually.', 'error');
        }
        document.body.removeChild(ta);
    }

    // ----------------------------------------------------------------
    // Confetti celebration (for achievements)
    // ----------------------------------------------------------------
    window.celebrate = function() {
        var colors = ['#E55000', '#FFC107', '#4CAF50', '#2196F3', '#9C27B0'];
        for (var i = 0; i < 40; i++) {
            setTimeout(function() {
                var el = document.createElement('div');
                el.className = 'confetti-piece';
                el.style.left = Math.random() * 100 + 'vw';
                el.style.background = colors[Math.floor(Math.random() * colors.length)];
                el.style.borderRadius = Math.random() > 0.5 ? '50%' : '2px';
                el.style.width = (Math.random() * 8 + 4) + 'px';
                el.style.height = (Math.random() * 8 + 4) + 'px';
                el.style.animationDuration = (Math.random() * 2 + 2) + 's';
                el.style.animationDelay = '0s';
                document.body.appendChild(el);
                setTimeout(function() { el.remove(); }, 4000);
            }, i * 50);
        }
    };

    // ----------------------------------------------------------------
    // Animate skill bars when they come into view
    // ----------------------------------------------------------------
    function animateSkillBars() {
        document.querySelectorAll('.skill-bar-fill').forEach(function(bar) {
            var rect = bar.getBoundingClientRect();
            if (rect.top < window.innerHeight && rect.bottom > 0) {
                var target = bar.getAttribute('data-target');
                if (target) bar.style.width = target + '%';
            }
        });
    }

    document.addEventListener('DOMContentLoaded', animateSkillBars);
    window.addEventListener('scroll', animateSkillBars);

    // ----------------------------------------------------------------
    // Init on page load
    // ----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function() {
        initUploadZone();

        // Auto-poll if on processing page
        var pollJobId = document.body.getAttribute('data-poll-job');
        if (pollJobId) {
            window.pollJob(pollJobId);
        }
    });

})();

// ----------------------------------------------------------------
// Utility functions (global)
// ----------------------------------------------------------------

function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '0s';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    var m = Math.floor(seconds / 60);
    var s = (seconds % 60).toFixed(0);
    return m + 'm ' + s + 's';
}

function formatDate(iso) {
    if (!iso) return '-';
    try {
        var d = new Date(iso);
        return d.toLocaleDateString() + ' ' +
               d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    } catch(e) {
        return iso.substring(0, 16);
    }
}

function phaseDisplay(phase) {
    return phase.replace(/_/g, ' ').replace(/\b\w/g, function(c) {
        return c.toUpperCase();
    });
}

function phaseClass(phase) {
    var map = {
        'stance': 'phase-stance', 'backlift': 'phase-backlift',
        'stride': 'phase-stride', 'downswing': 'phase-downswing',
        'impact': 'phase-impact', 'follow_through': 'phase-follow_through',
        'recovery': 'phase-recovery',
    };
    return map[phase] || '';
}

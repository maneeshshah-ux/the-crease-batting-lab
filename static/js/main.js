/**
 * the CREASE Batting Lab — Main JavaScript
 *
 * Job polling, PWA install, UI helpers, user interactions.
 */

(function() {
    'use strict';

    // ----------------------------------------------------------------
    // PWA Install Prompt (Android Chrome)
    // ----------------------------------------------------------------
    var deferredPrompt = null;
    var installBtn = document.getElementById('pwa-install-btn');

    window.addEventListener('beforeinstallprompt', function(e) {
        // Prevent default mini-infobar
        e.preventDefault();
        deferredPrompt = e;

        // Show install banner
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'block';

        if (installBtn) {
            installBtn.style.display = 'inline-block';
            installBtn.addEventListener('click', function() {
                banner.style.display = 'none';
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then(function(choice) {
                    deferredPrompt = null;
                });
            });
        }
    });

    window.addEventListener('appinstalled', function() {
        deferredPrompt = null;
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'none';
        console.log('📱 the CREASE installed successfully!');
    });

    // ----------------------------------------------------------------
    // Dismiss PWA banner
    // ----------------------------------------------------------------
    window.dismissPWA = function() {
        var banner = document.getElementById('pwa-install-banner');
        if (banner) banner.style.display = 'none';
    };

    // ----------------------------------------------------------------
    // Auto-dismiss flash alerts
    // ----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.alert-dismissible').forEach(function(alert) {
            setTimeout(function() {
                try {
                    var bs = bootstrap.Alert.getOrCreateInstance(alert);
                    bs.close();
                } catch(e) {}
            }, 6000);
        });
    });

    // ----------------------------------------------------------------
    // File input: show selected filename + size
    // ----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function() {
        var fileInput = document.getElementById('video');
        if (fileInput) {
            fileInput.addEventListener('change', function(e) {
                var target = e.target;
                var parent = target.closest('.mb-3');
                var hint = parent ? parent.querySelector('.form-text') : null;
                if (hint && target.files.length > 0) {
                    var f = target.files[0];
                    var sizeMB = (f.size / (1024 * 1024)).toFixed(1);
                    hint.textContent = f.name + ' (' + sizeMB + ' MB)';
                }
            });
        }
    });

    // ----------------------------------------------------------------
    // Confirm dialogs
    // ----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('[data-confirm]').forEach(function(el) {
            el.addEventListener('click', function(e) {
                if (!confirm(el.dataset.confirm)) e.preventDefault();
            });
        });
    });

})();

/**
 * Format duration in seconds.
 */
function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '0s';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    var m = Math.floor(seconds / 60);
    var s = (seconds % 60).toFixed(0);
    return m + 'm ' + s + 's';
}

/**
 * Format ISO date string.
 */
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

/**
 * Phase display name.
 */
function phaseDisplay(phase) {
    return phase.replace(/_/g, ' ').replace(/\b\w/g, function(c) {
        return c.toUpperCase();
    });
}

/**
 * Phase class for styling.
 */
function phaseClass(phase) {
    var map = {
        'stance': 'phase-stance', 'backlift': 'phase-backlift',
        'stride': 'phase-stride', 'downswing': 'phase-downswing',
        'impact': 'phase-impact', 'follow_through': 'phase-follow_through',
        'recovery': 'phase-recovery',
    };
    return map[phase] || '';
}

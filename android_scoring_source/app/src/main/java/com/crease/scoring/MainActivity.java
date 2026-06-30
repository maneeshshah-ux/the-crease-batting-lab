package com.crease.scoring;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.webkit.*;
import android.widget.*;
import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;

/**
 * the CREASE Cricket Scoring — Android WebView App
 *
 * A native Android wrapper that loads the Scoring App.
 * Loads from bundled assets for offline use.
 * Uses a virtual origin (https://crease.app/) so localStorage works properly.
 */
public class MainActivity extends Activity {

    private WebView webView;
    private LinearLayout loadingOverlay;
    private TextView loadingText;
    private ProgressBar loadingProgress;
    private LinearLayout errorView;
    private String appOrigin = "https://crease.app";
    private boolean isFirstLoad = true;
    private static final String PREFS_NAME = "crease_scoring_prefs";
    private static final String KEY_SERVER_URL = "server_url";
    private String lastError = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load saved server URL (for remote loading fallback)
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String savedOrigin = prefs.getString(KEY_SERVER_URL, appOrigin);
        if (savedOrigin != null && savedOrigin.startsWith("http")) {
            appOrigin = savedOrigin;
        }

        // Initialize views
        webView = findViewById(R.id.webview);
        loadingOverlay = findViewById(R.id.loading_overlay);
        loadingText = findViewById(R.id.loading_text);
        loadingProgress = findViewById(R.id.loading_progress);

        // Setup WebView
        setupWebView();

        // Load the app
        loadApp();
    }

    private void setupWebView() {
        // Enable JavaScript
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setAllowFileAccessFromFileURLs(false);
        settings.setAllowUniversalAccessFromFileURLs(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);

        // Main WebView client — intercepts asset requests and serves from bundle
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                // Serve bundled assets when the virtual origin is requested
                if (appOrigin.equals(uri.getScheme() + "://" + uri.getHost())) {
                    String path = uri.getPath(); // e.g. "/crease_logo.png"
                    if (path != null && path.length() > 1) {
                        String assetPath = path.substring(1); // remove leading "/"
                        try {
                            InputStream is = getAssets().open(assetPath);
                            String mime = "application/octet-stream";
                            if (assetPath.endsWith(".html")) mime = "text/html; charset=UTF-8";
                            else if (assetPath.endsWith(".png")) mime = "image/png";
                            else if (assetPath.endsWith(".json")) mime = "application/json";
                            else if (assetPath.endsWith(".js")) mime = "application/javascript";
                            else if (assetPath.endsWith(".css")) mime = "text/css";
                            else if (assetPath.endsWith(".svg")) mime = "image/svg+xml";
                            else if (assetPath.endsWith(".ico")) mime = "image/x-icon";
                            return new WebResourceResponse(mime, "UTF-8", is);
                        } catch (IOException e) {
                            // Asset not in bundle — let WebView handle (will fail gracefully)
                        }
                    }
                }
                return null; // Default handling
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                super.onPageFinished(view, url);
                hideLoading();
                isFirstLoad = false;
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request,
                    WebResourceError error) {
                super.onReceivedError(view, request, error);
                if (isFirstLoad) {
                    lastError = String.valueOf(error.getDescription());
                    showError("Failed to load the scoring app.\n\n" +
                             "Error: " + lastError);
                }
            }
        });

        // Enable hardware acceleration
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        }

        // Handle JavaScript dialogs (alert, confirm) so buttons work in the app
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onJsAlert(WebView view, String url, String message, JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                    .setTitle("the CREASE")
                    .setMessage(message)
                    .setPositiveButton("OK", (d, w) -> result.confirm())
                    .setCancelable(false)
                    .show();
                return true;
            }

            @Override
            public boolean onJsConfirm(WebView view, String url, String message, JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                    .setTitle("the CREASE")
                    .setMessage(message)
                    .setPositiveButton("Yes", (d, w) -> result.confirm())
                    .setNegativeButton("No", (d, w) -> result.cancel())
                    .setCancelable(false)
                    .show();
                return true;
            }
        });

        // Disable long-press context menu
        webView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                return true;
            }
        });
    }

    private void loadApp() {
        showLoading("Loading the CREASE Scoring...");
        try {
            // Read the HTML from assets
            InputStream is = getAssets().open("index.html");
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1) {
                baos.write(buf, 0, n);
            }
            is.close();
            String html = baos.toString("UTF-8");

            // Load with virtual origin — relative URLs like crease_logo.png
            // resolve to https://crease.app/crease_logo.png and are intercepted
            // by shouldInterceptRequest which serves them from assets.
            // This gives us a valid HTTPS origin for localStorage and JS.
            webView.loadDataWithBaseURL(appOrigin + "/", html, "text/html", "UTF-8", null);
        } catch (IOException e) {
            showError("Failed to read app data: " + e.getMessage());
        }
    }

    // Loading overlay
    private void showLoading(String message) {
        loadingOverlay.setVisibility(View.VISIBLE);
        loadingText.setText(message);
        loadingProgress.setIndeterminate(true);
    }

    private void hideLoading() {
        loadingOverlay.setVisibility(View.GONE);
    }

    private void showError(String message) {
        hideLoading();
        if (errorView == null) {
            errorView = new LinearLayout(this);
            errorView.setOrientation(LinearLayout.VERTICAL);
            errorView.setGravity(android.view.Gravity.CENTER);
            errorView.setPadding(40, 40, 40, 40);
            errorView.setBackgroundColor(0xff0A0A0A);

            TextView errorText = new TextView(this);
            errorText.setText(message);
            errorText.setTextColor(0xffC0C0C0);
            errorText.setTextSize(14);
            errorText.setGravity(android.view.Gravity.CENTER);
            errorView.addView(errorText);

            Button retryBtn = new Button(this);
            retryBtn.setText("Retry Connection");
            retryBtn.setTextColor(0xffF7F7F5);
            retryBtn.setBackgroundColor(0xffE55000);
            LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
            btnParams.topMargin = 30;
            retryBtn.setLayoutParams(btnParams);
            retryBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    loadApp();
                }
            });
            errorView.addView(retryBtn);

            Button settingsBtn = new Button(this);
            settingsBtn.setText("Change Server URL");
            settingsBtn.setTextColor(0xffC0C0C0);
            settingsBtn.setBackgroundColor(0xff1A1A1A);
            LinearLayout.LayoutParams sBtnParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
            sBtnParams.topMargin = 15;
            settingsBtn.setLayoutParams(sBtnParams);
            settingsBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showServerDialog();
                }
            });
            errorView.addView(settingsBtn);

            ((android.widget.FrameLayout) findViewById(R.id.webview_container)).addView(errorView);
        }
    }

    // Server URL configuration dialog
    private void showServerDialog() {
        final EditText input = new EditText(this);
        input.setText(appOrigin);
        input.setSelectAllOnFocus(true);
        input.setHint("e.g. https://example.com/scoring/");
        input.setHintTextColor(0xff888888);

        new android.app.AlertDialog.Builder(this)
            .setTitle("Server URL")
            .setMessage("Enter the URL where the Scoring App is hosted (or 'local' for bundled app):")
            .setView(input)
            .setPositiveButton("Connect", new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    String url = input.getText().toString().trim().toLowerCase();
                    if ("local".equals(url)) {
                        appOrigin = "https://crease.app";
                    } else if (!url.isEmpty()) {
                        appOrigin = url;
                    }
                    SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
                    prefs.edit().putString(KEY_SERVER_URL, appOrigin).apply();
                    if (errorView != null) {
                        ((android.widget.FrameLayout) findViewById(R.id.webview_container))
                            .removeView(errorView);
                        errorView = null;
                    }
                    loadApp();
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    // Back button navigation
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            if (webView.canGoBack()) {
                webView.goBack();
                return true;
            }
            // Single-page app: navigate within the app via JavaScript
            // Step 1: Try closing any open modals (saved sessions, innings summary, etc.)
            // Step 2: If view isn't setup, go back to setup
            // Step 3: If all else fails, close the app
            webView.evaluateJavascript(
                "(function() { " +
                "  try { " +
                "    if (typeof closeAllModals === 'function') { " +
                "      var modals = document.querySelectorAll('.modal-overlay').length; " +
                "      if (modals > 0) { closeAllModals(); return 'navigated'; } " +
                "    } " +
                "    if (typeof S !== 'undefined') { " +
                "      if (S().view === 'match' || S().view === 'pairselect' || S().view === 'summary' || S().view === 'scorecard') { " +
                "        S().view = 'setup'; render(); " +
                "        return 'navigated'; " +
                "      } " +
                "      if (S().view === 'setup') { " +
                "        // Check if saved sessions or session viewer is showing (inline render) " +
                "        var app = document.getElementById('app'); " +
                "        if (app && app.innerHTML.indexOf('Saved Sessions') !== -1) { " +
                "          S().view = 'setup'; render(); " +
                "          return 'navigated'; " +
                "        } " +
                "      } " +
                "    } " +
                "  } catch(e) { } " +
                "  return 'exit'; " +
                "})()",
                new ValueCallback<String>() {
                    @Override
                    public void onReceiveValue(String value) {
                        if (value == null || "\"exit\"".equals(value)) {
                            MainActivity.this.finish();
                        }
                    }
                }
            );
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    // Menu for settings
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, "Server Settings");
        menu.add(0, 2, 1, "Refresh");
        menu.add(0, 3, 2, "About");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case 1:
                showServerDialog();
                return true;
            case 2:
                webView.reload();
                return true;
            case 3:
                new android.app.AlertDialog.Builder(this)
                    .setTitle("the CREASE Scoring")
                    .setMessage("v1.1.0\n\n" +
                               "Indoor cricket match scoring app — all 13 bug fixes.\n\n" +
                               "Where every cricketer gets better.\n" +
                               "© 2026 the CREASE")
                    .setPositiveButton("OK", null)
                    .show();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    // Handle configuration changes (e.g., keyboard show/hide)
    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        // WebView handles its own config changes
    }

    // Save state
    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        webView.saveState(outState);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        webView.restoreState(savedInstanceState);
    }
}

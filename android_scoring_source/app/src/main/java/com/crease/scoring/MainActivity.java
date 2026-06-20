package com.crease.scoring;

import android.app.Activity;
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
import java.io.File;
import java.net.URI;

/**
 * the CREASE Cricket Scoring — Android WebView App
 *
 * A native Android wrapper that loads the Scoring App PWA.
 * Default URL: https://the-crease-batting-lab.onrender.com/scoring/
 */
public class MainActivity extends Activity {

    private WebView webView;
    private LinearLayout loadingOverlay;
    private TextView loadingText;
    private ProgressBar loadingProgress;
    private LinearLayout errorView;
    private String serverUrl = "https://the-crease-batting-lab.onrender.com/scoring/";
    private boolean isFirstLoad = true;
    private static final String PREFS_NAME = "crease_scoring_prefs";
    private static final String KEY_SERVER_URL = "server_url";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load saved server URL
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        serverUrl = prefs.getString(KEY_SERVER_URL, serverUrl);

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
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);

        // Main WebView client
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
                super.onPageStarted(view, url, favicon);
                if (isFirstLoad) {
                    showLoading("Loading the CREASE Scoring...");
                }
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
                    showError("Cannot connect to server.\n\n" +
                             "Make sure you have internet access.\n\n" +
                             "Server URL: " + serverUrl);
                }
            }

            @Override
            public void onReceivedSslError(WebView view, SslErrorHandler handler,
                    android.net.http.SslError error) {
                handler.proceed(); // Allow valid SSL certs
            }
        });

        // Enable hardware acceleration
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        }

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
        webView.loadUrl(serverUrl);
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
        input.setText(serverUrl);
        input.setSelectAllOnFocus(true);
        input.setTextColor(0xffC0C0C0);
        input.setHint("e.g. https://example.com/scoring/");
        input.setHintTextColor(0xff888888);

        new android.app.AlertDialog.Builder(this)
            .setTitle("Server URL")
            .setMessage("Enter the URL where the Scoring App is hosted:")
            .setView(input)
            .setPositiveButton("Connect", new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    String url = input.getText().toString().trim();
                    if (!url.isEmpty()) {
                        serverUrl = url;
                        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
                        prefs.edit().putString(KEY_SERVER_URL, serverUrl).apply();
                        if (errorView != null) {
                            ((android.widget.FrameLayout) findViewById(R.id.webview_container))
                                .removeView(errorView);
                            errorView = null;
                        }
                        loadApp();
                    }
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    // Back button navigation
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack()) {
            webView.goBack();
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
                    .setMessage("v1.0.0\n\n" +
                               "Indoor cricket match scoring app.\n\n" +
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

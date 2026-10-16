
// --- VisiOS: defaults that make Firefox predictable for a vision agent ---
// Stable screen: no blinking caret, no smooth scrolling, no UI animations.
pref("ui.caretBlinkTime", 0);
pref("general.smoothScroll", false);
pref("ui.prefersReducedMotion", 1);
pref("toolkit.cosmeticAnimations.enabled", false);
// Software rendering: reliable inside a container without a GPU.
pref("gfx.webrender.software", true);
pref("layers.acceleration.disabled", true);
// No interruptions: crash restore, safe-mode prompt, close warnings, tours.
pref("browser.sessionstore.resume_from_crash", false);
pref("toolkit.startup.max_resumed_crashes", -1);
pref("browser.tabs.warnOnClose", false);
pref("browser.tabs.warnOnCloseOtherTabs", false);
pref("browser.warnOnQuit", false);
pref("browser.shell.checkDefaultBrowser", false);
pref("browser.startup.homepage_override.mstone", "ignore");
pref("browser.aboutwelcome.enabled", false);
pref("browser.aboutConfig.showWarning", false);
pref("browser.uitour.enabled", false);
pref("browser.translations.automaticallyPopup", false);
pref("browser.tabs.firefox-view", false);
pref("browser.ml.chat.enabled", false);
pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
// Reject cookie banners automatically when Firefox knows how.
pref("cookiebanners.service.mode", 2);
pref("cookiebanners.service.mode.privateBrowsing", 2);
// No autoplaying audio or video.
pref("media.autoplay.default", 5);
// New windows and tabs open on the search page.
pref("browser.newtabpage.enabled", true);
pref("browser.startup.page", 1);

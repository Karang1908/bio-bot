---
name: Bio-Bot Studio
description: Control room for one self-learning brain and the bodies it lives in.
colors:
  panel: "rgba(15, 17, 22, 0.9)"
  panel-solid: "#0F1116"
  page: "#0B0D12"
  line: "rgba(255, 255, 255, 0.1)"
  line-strong: "rgba(255, 255, 255, 0.18)"
  text: "#F4F4F5"
  muted: "#A1A1AA"
  faint: "#8B8B94"
  good: "#4ADE80"
  warn: "#FBBF24"
  bad: "#F87171"
  focus: "#93C5FD"
  humanoid: "#F59E0B"
  dog: "#2DD4BF"
  drone: "#60A5FA"
  fly: "#C084FC"
  car: "#F472B6"
  neuron-rest: "#4D6186"
  neuron-active: "#FFCC61"
typography:
  body:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  small:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.45
  title:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1.3
rounded:
  control: "10px"
  surface: "14px"
  pill: "999px"
spacing:
  edge: "16px"
  gap: "12px"
  card: "16px"
components:
  button:
    backgroundColor: "rgba(255, 255, 255, 0.06)"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "0 12px"
  button-primary:
    backgroundColor: "{colors.text}"
    textColor: "{colors.page}"
    rounded: "{rounded.control}"
  chip:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.pill}"
    height: "36px"
  card:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.surface}"
    padding: "16px"
  command-box:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "16px"
    height: "52px"
---

# Design System: Bio-Bot Studio

## Overview

**Creative North Star: "The World Is the Screen"**

The realistic 3D world fills the whole window. Controls float over it as small dark panels, and only what you are working with is open: the body you clicked or the Brain (in the right-hand dock), the World menu, the map. Everything is written in plain words ("Stand", "Relax", "Explore", "Motor power", "Put back") so someone new can use it without a tutorial, and a projector audience can follow along.

The user chose this direction ("World-first, minimal", with a readable font and one colour per body) after finding the earlier pixel-font Cracktro design too hard to understand. It replaces that design completely.

**Key Characteristics:**
- The 3D world is never hidden by a permanent sidebar.
- One colour per body, used on its chip, inspector, sliders, map marker and ground ring.
- Getting around never needs a mouse gesture you have to discover: every camera move has a button, a key, and a place on the map.
- Panels open on demand and close with ×, Esc, or by clicking the same chip again.
- The brain view shows real data only: neuron positions from MaleCNS and live activity from the running network.

## Colors

Neutral dark panels over a bright, realistic scene; colour is reserved for identity (bodies) and meaning (status, activity).

### Body colours
- **Humanoid Amber** (#F59E0B), **Dog Teal** (#2DD4BF), **Drone Blue** (#60A5FA), **Fly Violet** (#C084FC), **Car Rose** (#F472B6): each body's chip icon, selected-chip border and tint, inspector badge and state word, sliders, map marker, and the ring under it in the world. Car Rose is deliberately not Bad Red, so a car chip never reads as an error.

### Status
- **Good Green** (#4ADE80): live status, the brain "Listening" switch, finished roadmap steps.
- **Warn Amber** (#FBBF24): paused.
- **Bad Red** (#F87171): fallen or crashed bodies, offline, errors.

### Brain
- **Resting Neuron** (#4D6186 at low alpha) and **Active Neuron** (#FFCC61): the 3D neuron cloud and the group activity bars.

### Neutral
- **Panel** (rgba(15,17,22,0.9)) with a 14px blur so text stays legible over the bright scene; **Panel Solid** (#0F1116) for the dock, menus and dialogs; **Text** (#F4F4F5); **Muted** (#A1A1AA, 7.4:1 on the panel) for secondary text; **Faint** (#8B8B94, 5.3:1) for placeholders and credits.

### Named Rules
**The Identity Rule.** A body's colour means "this is that body", everywhere. Never use a body colour for status or decoration.

**The Real Data Rule.** Nothing in the brain view is illustrative. If a number or a glow is on screen, the running brain produced it.

## Typography

**Font:** the platform system UI font (SF Pro on the Mac), chosen for legibility; no web font is loaded.

### Hierarchy
- **Title** (600, 16px): card and menu headings.
- **Body** (400, 14px, line-height 1.45): everything else; the command box input is 15px.
- **Small** (400, 12.5px): help text, descriptions, legends, credits (11.5px).
- Numbers in stats and bars use tabular figures.

## Layout

The world canvas fills the window. Floating layers, all 16px from the edges:
- **Top left:** brand, live status pill and the map switch (Sandbox / Open world); under it, the lineup: one chip per body (name + one-word state + × to remove) and an "Add body" menu (one body, or all).
- **Top right:** Pause/Resume, Reset, World, Brain, Help (?).
- **Bottom left:** the navigator (232px): a live map you can click, camera buttons (turn, zoom, top view, start view) and one button per place.
- **Bottom centre:** the command box (max 680px), between the navigator and the dock, with example commands for the current map or the latest reply above it.
- **Right:** the dock (400px, full height) with two tabs, Body (the inspector) and Brain. When it is open, the world, top bar and command box shift left so nothing is covered.

Below 820px: action buttons show icons only (with accessible names), the brand name hides and the map switch moves into the World menu, chips and examples scroll horizontally, the navigator collapses to a "Map" button, and the dock becomes a bottom sheet (64% of the height) that hides the command box while open.

## Elevation & Depth

Two levels. The world is the base. Every floating control sits one level above it on a translucent dark panel with a 1px white/10% edge and a soft offset shadow (`0 10px 30px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.25)`). The dock is opaque because it is a workspace, not an overlay; the help dialog dims the world behind it. Panels enter with a short rise or slide (200–240ms, ease-out); with reduced motion requested, nothing animates.

## Shapes

Rounded but not soft: 10px controls, 14px cards and panels, 16px command box, fully round chips and suggestion pills.

## Components

### Buttons
- **Default:** 34px tall, 10px radius, white/6% fill, 1px white/18% edge, icon + label.
- **Primary / on:** light fill with dark text (the open Brain button, the pressed Follow button, the active segment).
- **Ghost (top bar):** panel fill with blur and shadow so they read over the scene.
- **Disabled:** 50% opacity with a reason in the label or title.
- **Focus:** 2px light-blue outline, 2px offset, on every control.

### Body chips
Round pill: body icon in the body's colour, name in bold, one-word state ("Standing", "Exploring", "Flying", "Driving", "Swimming", "Relaxed", or "Fallen" / "Flipped" in red), and a small × that removes the body from the scene (hidden when it is the last one). Selected: border and 18% tint in the body's colour. Clicking selects the body and opens its inspector. The dashed "Add body" chip opens a menu of the bodies not in the scene, plus "Add all".

### Inspector (Body tab of the dock)
Badge, name and model; state line ("Standing · 0.79 m up · 0.0 m/s"); a four-way segmented control (Stand or Hover or Brake / Relax / Explore / Take control, Drive it or Fly it) with one line explaining the selected mode. For the car and drone, a hold-to-repeat direction pad (speed or height in the middle), Up/Down for the drone, and "Drive itself to" / "Fly itself to" place buttons. Then the Motor power slider; Put back, Find and Follow; the Motors list (one slider per actuator showing the live joint angle, enabled only in Take control; hovering a row highlights that part in 3D; a filter appears above 12 motors); a collapsible Sensors list with live values; mass, parts, joints and motors; the model's credit.

### Navigator
A 208px map drawn from the real scene: terrain shaded by height, the lake, roads, the house and furniture, tree trunks and place names, all at their live positions. Bodies appear as dots in their colour with a heading tick (the car is a small oriented rectangle), and the apple is a red dot with a white ring. The camera is a white ring plus a translucent wedge showing where it looks. Click or drag to move the camera; click a body to select it and fly to it. Under the map: turn left/right and zoom in/out (hold to repeat), top view, start view, then one pill per place.

### Command box
One input ("Tell the bodies what to do…") and a round send button; up and down arrows recall history. Above it, either up to five example commands that work on the current map with the current bodies ("Drive to the lake" only when the car is there), or the latest reply (with the command that produced it, and a dismiss button). Errors show a red icon.

### World menu
A one-line description of the map; Map switch (narrow screens only); Apple location (select, from the map's own spots); Speed (0.5× / 1× / 2×); Top view; Show collision shapes; and a footer with physics speed and frame rate.

### Help dialog
Opened with the ? button or key. Four groups (Look around, Bodies, Drive and fly, Everything else), each row a key or gesture in a key cap and what it does. Esc, the × or a click outside closes it.

### Brain tab (signature component)
The honest subtitle "Real fruit-fly wiring (MaleCNS) · not trained yet"; a Listening switch; the 3D neuron cloud (brain on top, nerve cord below, auto-rotating, drag to rotate, resting points dim blue, active points glowing gold); three stats (neurons, connections, active now); seven plain-language groups (Senses → Muscles) with activity bars, where hovering a group tints its neurons in 3D; a note on how to read it; and the training roadmap with done / next / locked steps.

## Do's and Don'ts

### Do:
- **Do** keep the world visible: open panels only on demand and close them easily.
- **Do** write labels as plain actions and states a newcomer understands.
- **Do** use each body's colour only for that body.
- **Do** label anything unfinished honestly ("not trained yet", "coming with stage 1").
- **Do** keep every control reachable by keyboard with a visible focus ring.

### Don't:
- **Don't** add a permanent sidebar or dashboard grid over the world.
- **Don't** use jargon in the main UI ("babble", "rtf", "stage 1a"); keep it in the docs.
- **Don't** show brain activity, learning curves or numbers the system did not produce.
- **Don't** tint or colour-grade the 3D world.

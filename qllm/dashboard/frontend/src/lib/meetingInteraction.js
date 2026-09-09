import { meetingReply } from "./researchStudios.js";
import { clampSample, KINETIC_STUDIOS } from "./kineticsBench.js";
import { captureResultCard, heldResultCard } from "./resultCards.js";
import { boundedLaunchAngle } from "./launcherInteraction.js";
import { newConversationTrial, editConversationTrial, completeConversationTrial } from "./conversationTrial.js";
import { LANDING_TARGETS, initialLandingChallenge, landingTarget, recordLandingShot, landChallengeShot, landingObservation } from "./landingChallenge.js";

export const ROOM_ACTIONS = [
  { id: "greet", label: "Talk to researcher", icon: "send", hint: "Say hello" },
  {
    id: "coffee",
    label: "Have a coffee",
    icon: "coffee",
    hint: "Take a moment",
  },
  {
    id: "paper",
    label: "Take working note",
    icon: "paper",
    hint: "Read together",
  },
  {
    id: "board",
    label: "Use shared board",
    icon: "mathematics",
    hint: "Think out loud",
  },
  {
    id: "experiment",
    label: "Try an experiment",
    icon: "code",
    hint: "Change one thing",
  },
];
const OFFER_REPLY = "Here’s the working note. Take it from my hand when you’re ready, or leave it on the table. This is an illustrative note, not a published paper.";

export function initialMeeting(studio) {
  return {
    messages: [
      {
        role: "agent",
        text: `Come on in. Click me to say hello, take the note from the table, or bring an idea to the board. This is an interactive, scripted preview — no research model is connected.`,
      },
    ],
    draft: "",
    experiment: "",
    saved: false,
    material: null,
    quote: "",
    board: { question: studio.title, assumption: "", test: "", strokes: [] },
    illustration: {},
    gesture: { type: "idle", id: 0 },
    noteHeld: false,
    noteOffered: false,
    noteRevision: 0,
    resultCards: [],
    heldResultId: null,
    selectedResultId: null,
    resultRevision: 0,
    launcherActive: false,
    launcherChallenge: initialLandingChallenge(),
    visited: [],
  };
}

// Remount restores the visit's current action as settled, never as a new gesture.
export function initialGestureClock(id = 0) {
  return { elapsed: 0, id, start: 0, instant: true, age: 10 };
}

// Equipment gets the working surface; explicit conversation intents reveal chat.
// Calculation and scrub events never move keyboard focus or change this choice.
export function meetingRoomFocus(
  focused,
  event,
  previousMaterial,
  nextMaterial,
  newOffer = false,
) {
  if (newOffer) return true;
  if (event.type === "trial-open") return false;
  if (event.type === "launcher" || event.type === "landing-challenge") return true;
  if (event.type === "result-card")
    return ["collect", "carry", "store"].includes(event.action);
  if (event.type === "note" && event.action === "read") return false;
  if (event.type === "note" && ["take", "offer", "decline"].includes(event.action)) return true;
  if (
    event.type === "close-material" &&
    ["paper", "result"].includes(previousMaterial)
  )
    return true;
  if (event.type === "interact" && event.id === "paper")
    return nextMaterial !== "paper";
  if (
    event.type === "quote" ||
    (event.type === "interact" && event.id === "greet")
  )
    return false;
  if (
    nextMaterial === "experiment" &&
    (previousMaterial !== "experiment" ||
      (event.type === "interact" && event.id === "experiment"))
  )
    return true;
  if (event.type === "send" || event.type === "close-material") return false;
  return focused;
}

export function meetingEvent(previous, studio, event) {
  const state = { ...initialMeeting(studio), ...previous };
  if (["interact", "note", "result-card", "quote", "send"].includes(event.type))
    state.launcherActive = false;
  const gesture = (type) => ({ type, id: (state.gesture?.id || 0) + 1 });
  const visit = (id) => [...new Set([...state.visited, id])];
  if (event.type === "trial-open") {
    const trial = newConversationTrial(studio.id, state.illustration.parameter);
    if (!trial) return state;
    return { ...state, messages: [...state.messages, {
      role: "calculation", text: "Change one input while we talk. This is a built-in browser experiment, separate from your research proposal.", trial,
    }] };
  }
  if (event.type === "trial-input") {
    const message = state.messages[event.index];
    if (!Number.isInteger(event.index) || !message?.trial) return state;
    const trial = editConversationTrial(message.trial, event.parameter);
    if (trial === message.trial) return state;
    return { ...state, messages: state.messages.map((item, index) => index === event.index ? { ...item, trial } : item) };
  }
  if (event.type === "trial-status") {
    if (!Number.isInteger(event.index) || !state.messages[event.index]?.trial || typeof event.error !== "string") return state;
    return { ...state, messages: state.messages.map((message, index) => index === event.index
      ? { ...message, trial: { ...message.trial, error: event.error } } : message) };
  }
  if (event.type === "landing-challenge") {
    if (studio.id !== "physics") return state;
    const challenge = state.launcherChallenge;
    let targetId;
    if (event.action === "start") targetId = challenge.lastTargetId;
    else if (event.action === "stop") targetId = null;
    else if (event.action === "next") {
      const index = LANDING_TARGETS.findIndex((t) => t.id === challenge.targetId);
      targetId = LANDING_TARGETS[(index + 1) % LANDING_TARGETS.length].id;
    } else return state;
    if (targetId !== null && !landingTarget(targetId)) return state;
    return { ...state, launcherChallenge: { ...challenge, targetId, lastTargetId: targetId || challenge.lastTargetId } };
  }
  if (event.type === "challenge-landed") {
    if (studio.id !== "physics" || event.flightId !== state.illustration.flightId) return state;
    const challenge = landChallengeShot(state.launcherChallenge, event.flightId);
    if (challenge === state.launcherChallenge) return state;
    const shot = challenge.shots.find((s) => s.flightId === event.flightId);
    return { ...state, launcherChallenge: challenge, messages: [...state.messages, { role: "agent", text: `${landingObservation(shot)} ${shot.hit ? "Can you find another angle that reaches the same pad?" : "What would you change for the next shot?"} This is scripted challenge feedback.` }] };
  }
  if (event.type === "launcher") {
    if (studio.id !== "physics") return state;
    if (event.action === "open")
      return {
        ...state,
        launcherActive: true,
        material: null,
        noteHeld: false,
        noteOffered: false,
        noteRevision: state.noteRevision + Number(state.noteHeld || state.noteOffered),
        heldResultId: null,
        resultRevision:
          state.resultRevision + Number(state.heldResultId !== null),
      };
    if (event.action === "close") return { ...state, launcherActive: false };
    if (event.action === "aim" && Number.isFinite(event.angle))
      return {
        ...state,
        illustration: {
          ...state.illustration,
          parameter: boundedLaunchAngle(event.angle),
        },
      };
    return state;
  }
  if (event.type === "patch") return { ...state, ...event.value };
  if (event.type === "board")
    return { ...state, board: { ...state.board, ...event.value } };
  if (event.type === "close-material") return { ...state, material: null };
  if (event.type === "result-card") {
    if (event.action === "collect") {
      const captured = captureResultCard(event.illustration);
      if (!captured || captured.modelId !== studio.id) return state;
      const card = Object.freeze({
        ...captured,
        id: state.resultCards.length + 1,
      });
      return {
        ...state,
        resultCards: [...state.resultCards, card],
        heldResultId: card.id,
        selectedResultId: card.id,
        noteHeld: false,
        noteOffered: false,
        noteRevision: state.noteRevision + Number(state.noteHeld || state.noteOffered),
        resultRevision: state.resultRevision + 1,
        material: null,
        gesture: gesture("paper"),
      };
    }
    const card = event.id
      ? state.resultCards.find((item) => item.id === event.id)
      : heldResultCard(state);
    if (!card) return state;
    if (event.action === "carry")
      return {
        ...state,
        heldResultId: card.id,
        selectedResultId: card.id,
        noteHeld: false,
        noteOffered: false,
        noteRevision: state.noteRevision + Number(state.noteHeld || state.noteOffered),
        resultRevision: state.resultRevision + 1,
        material: null,
        board: {
          ...state.board,
          pinnedResultIds: (state.board.pinnedResultIds || []).filter(
            (id) => id !== card.id,
          ),
        },
      };
    if (event.action === "read")
      return { ...state, selectedResultId: card.id, material: "result" };
    if (event.action === "store" && state.heldResultId === card.id)
      return {
        ...state,
        heldResultId: null,
        resultRevision: state.resultRevision + 1,
        material: null,
      };
    if (event.action === "pin" && state.heldResultId === card.id)
      return {
        ...state,
        heldResultId: null,
        resultRevision: state.resultRevision + 1,
        material: "board",
        gesture: gesture("board"),
        board: {
          ...state.board,
          pinnedResultIds: [
            ...new Set([...(state.board.pinnedResultIds || []), card.id]),
          ],
        },
        messages: [
          ...state.messages,
          {
            role: "agent",
            text: `Card ${card.id} is on our board. It preserves the browser calculation you collected. What would you change next? We can discuss its assumptions; this is still a scripted preview.`,
          },
        ],
      };
    return state;
  }
  if (event.type === "note") {
    if (event.action === "offer") {
      if (state.noteHeld) return { ...state, material: "paper" };
      if (state.noteOffered) return state;
      return {
        ...state, noteOffered: true, noteRevision: state.noteRevision + 1,
        material: null,
        messages: [...state.messages, { role: "agent", text: OFFER_REPLY }],
      };
    }
    if (event.action === "decline")
      return state.noteOffered ? { ...state, noteOffered: false, noteRevision: state.noteRevision + 1 } : state;
    if (!["take", "read", "return"].includes(event.action)) return state;
    const held = event.action !== "return";
    return {
      ...state,
      noteHeld: held,
      noteOffered: false,
      noteRevision: state.noteRevision + Number(held !== state.noteHeld || state.noteOffered),
      ...(held
        ? {
            heldResultId: null,
            resultRevision:
              state.resultRevision + Number(state.heldResultId !== null),
          }
        : {}),
      material:
        event.action === "read"
          ? "paper"
          : state.material === "paper"
            ? null
            : state.material,
      visited: held ? visit("paper") : state.visited,
    };
  }
  if (event.type === "quote")
    return {
      ...state,
      quote: event.text.slice(0, 1200),
      quoteKind: event.kind === "observation" ? "observation" : "passage",
    };
  if (event.type === "scrub")
    return KINETIC_STUDIOS.includes(state.illustration.result?.id)
      ? {
          ...state,
          illustration: {
            ...state.illustration,
            sampleIndex: clampSample(event.index),
          },
        }
      : state;
  if (event.type === "result")
    return {
      ...state,
      launcherChallenge: studio.id === "physics" ? recordLandingShot(state.launcherChallenge, event.result, event.targetId, (state.illustration.flightId || 0) + 1) : state.launcherChallenge,
      illustration: {
        ...state.illustration,
        previousResult: state.illustration.result || null,
        result: event.result,
        sampleIndex: 60,
        flightId: (state.illustration.flightId || 0) + 1,
      },
      gesture: gesture("experiment"),
      visited: visit("experiment"),
      messages: Number.isInteger(event.trialIndex) && state.messages[event.trialIndex]?.trial
        ? state.messages.map((message, index) => index === event.trialIndex
          ? { ...message, trial: completeConversationTrial(message.trial, event.result) }
          : message)
        : [
        ...state.messages,
        {
          role: "calculation",
          text: `Browser illustration complete: ${event.result.points.length} values. Comparison ${event.result.parameter}; baseline ${event.result.baselineParameter}. Synthetic inputs only — this is not a test of your research proposal.`,
        },
      ],
    };
  if (event.type === "interact") {
    const id = event.id;
    if (!ROOM_ACTIONS.some((action) => action.id === id)) return state;
    const replies = {
      greet: `Hi, I’m ${studio.name.split(" ")[0]}. What are you curious about? Try an idea on the shared board, or ask about a passage in the note.`,
      coffee:
        "A little room to think. What is the one assumption you would most like to question?",
      paper:
        "The working note is yours to carry. Read it when you like, point to a passage, or put it back on the coffee table.",
      board:
        "The board is ours. Write a question, sketch a connection, and choose what we should try next. Your edits stay here during this visit.",
      experiment:
        "Let’s change one thing. The built-in illustration runs in your browser; your research idea stays a separate draft.",
    };
    return {
      ...state,
      gesture: gesture(id),
      visited: visit(id),
      ...(id === "paper"
        ? {
            noteHeld: true,
            noteOffered: false,
            noteRevision: state.noteRevision + Number(!state.noteHeld),
            heldResultId: null,
            resultRevision:
              state.resultRevision + Number(state.heldResultId !== null),
          }
        : {}),
      material:
        id === "paper"
          ? state.noteHeld
            ? "paper"
            : null
          : ["board", "experiment"].includes(id)
            ? id
            : null,
      messages: [...state.messages, { role: "agent", text: replies[id] }],
    };
  }
  if (event.type === "send") {
    const text = (event.text ?? state.draft).trim();
    if (!text) return state;
    const reply = meetingReply(studio, text);
    const offering = !state.quote && reply.action === "paper" && !state.noteHeld;
    let answer = offering ? OFFER_REPLY : reply.text;
    if (state.quote)
      answer =
        state.quoteKind === "observation"
          ? `Looking at this saved observation: “${state.quote}” What would you expect if we changed just one input? Keep the comparison conditions fixed and distinguish what the toy calculates from what a real experiment would establish. This is a scripted prompt about the browser toy, not an evaluation of research evidence.`
          : `About “${state.quote}”: separate what this passage assumes from what it actually establishes. What observation would contradict it? This is a scripted discussion prompt, not a source evaluation.`;
    else if (
      /assumption|challenge/i.test(text) &&
      state.board.assumption.trim()
    )
      answer = `On our board: “${state.board.assumption}”. Try writing one counterexample and the observation that would distinguish it. I’ve kept your wording unchanged; this is a scripted prompt, not an assessment of the idea.`;
    return {
      ...state,
      draft: event.text === undefined ? "" : state.draft,
      quote: "",
      messages: [
        ...state.messages,
        { role: "human", text, quote: state.quote },
        { role: "agent", text: answer },
      ],
      gesture: gesture("talk"),
      material: offering ? null : state.quote ? state.material : reply.action || state.material,
      ...(reply.action === "experiment"
        ? { experiment: state.experiment || text, saved: false }
        : {}),
      ...(offering
        ? { noteOffered: true, noteRevision: state.noteRevision + Number(!state.noteOffered) }
        : !state.quote && reply.action === "paper"
        ? {
            noteHeld: true,
            noteOffered: false,
            noteRevision: state.noteRevision + Number(!state.noteHeld),
            heldResultId: null,
            resultRevision:
              state.resultRevision + Number(state.heldResultId !== null),
          }
        : {}),
    };
  }
  return state;
}

export function advanceGestureClock(clock, dt, id, paused, reducedMotion) {
  const elapsed =
    clock.elapsed +
    (paused || reducedMotion ? 0 : Math.max(0, Math.min(dt, 0.05)));
  const changed = id !== clock.id;
  const start = changed ? elapsed : clock.start;
  // Manual pause freezes existing motion. New actions while paused, and reduced
  // motion actions, settle once; resuming cannot replay a completed gesture.
  const instant = reducedMotion || (changed ? paused : clock.instant);
  return { elapsed, id, start, instant, age: instant ? 10 : elapsed - start };
}

export { WALK_OBSTACLES, stepVisitor } from "./studioWalk.js";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import Icon from "./WorldIcon.jsx";
import LocalIllustration from "./LocalIllustration.jsx";
import InteractiveStudio from "./InteractiveStudio.jsx";
import SharedBoard from "./SharedBoard.jsx";
import ResultCard from "./ResultCard.jsx";
import ConversationTrial from "./ConversationTrial.jsx";
import StudioDialog from "./StudioDialog.jsx";
import { calculationEvent, calculationForRoom } from "../lib/conversationTrial.js";
import { heldResultCard, resultCardView } from "../lib/resultCards.js";
import {
  initialMeeting,
  meetingEvent,
  ROOM_ACTIONS,
  advanceGestureClock,
  meetingRoomFocus,
} from "../lib/meetingInteraction.js";
import { flightPresentation } from "../lib/projectileToy.js";
import { KINETIC_STUDIOS } from "../lib/kineticsBench.js";
import { createStudioCalculation } from "../lib/studioCalculation.js";
import { DEMO_MODELS } from "../lib/studioSimulation.js";
import { loadResearchRobot } from "../lib/researchRobot.js";
import { workingNote } from "../lib/workingNote.js";
import { conversationEntries, roomConversationMode, conversationMaterialShortcut } from "../lib/conversationView.js";
import "../interactive-meeting.css";
import "../room-inspection.css";
import "../first-person-hud.css";
import "../room-dialogue.css";

export default function ResearchSession({
  studio,
  onClose,
  paused,
  reducedMotion,
  session,
  onSessionChange,
}) {
  const dialogRef = useRef(null),
    transcript = useRef(null),
    room = useRef(null),
    current = useRef(null);
  const initial = useRef(initialMeeting(studio)),
    [localPaused, setLocalPaused] = useState(false),
    [roomFocused, setRoomFocused] = useState(
      session?.material === "experiment",
    );
  const calculator = useRef(null),
    dispatchRef = useRef(null);
  const [calculationState, setCalculationState] = useState({
    busy: false,
    error: "",
  });
  const [robotAsset, setRobotAsset] = useState({ ready: false, template: null, failed: false });
  const [studioView, setStudioView] = useState("overview");
  const [fullConversation, setFullConversation] = useState(false);
  useEffect(() => {
    if (studioView === "first-person") setRoomFocused(true);
  }, [studioView]);
  useEffect(() => {
    let active = true;
    loadResearchRobot(studio.id).then(
      (template) => { if (active) setRobotAsset({ ready: true, template, failed: false }); },
      () => { if (active) setRobotAsset({ ready: true, template: null, failed: true }); },
    );
    return () => { active = false; };
  }, [studio.id]);
  useEffect(() => {
    const owner = createStudioCalculation({
      createWorker: () =>
        new Worker(
          new URL("../lib/studioSimulation.worker.js", import.meta.url),
          { type: "module" },
        ),
      onState: (state) => {
        setCalculationState(state);
        if (Number.isInteger(state.context?.trialIndex))
          dispatchRef.current({ type: "trial-status", index: state.context.trialIndex, error: state.error });
      },
      onResult: (result, context) => dispatchRef.current(calculationEvent(result, context)),
    });
    calculator.current = owner;
    return () => {
      owner.dispose();
      calculator.current = null;
    };
  }, [studio.id]);
  const calculation = {
    ...calculationForRoom(calculationState),
    runTrial: (index) => {
      const trial = current.current.messages[index]?.trial;
      if (trial && trial.studioId === studio.id)
        calculator.current?.start(studio.id, trial.parameter, { trialIndex: index });
    },
    run: () =>
      calculator.current?.start(
        studio.id,
        current.current.illustration.parameter ??
          DEMO_MODELS[studio.id]?.initial,
        studio.id === "physics" ? current.current.launcherChallenge.targetId : null,
      ),
  };
  const value = { ...initial.current, ...session };
  const conversationMode = roomConversationMode(studioView, roomFocused, value, fullConversation);
  const roomConversationActive = conversationMode !== null;
  const roomDialogue = conversationMode === "dialogue";
  const carriedResult = heldResultCard(value);
  const selectedResult = value.resultCards.find(
    (card) => card.id === value.selectedResultId,
  );
  current.current = value;
  const flightRef = useRef(null),
    motionRef = useRef({});
  motionRef.current = { paused: paused || localPaused, reducedMotion };
  // One semantic flight clock publishes an immutable sample before scene/SVG consumers.
  // It survives material switches and works even when WebGL is unavailable.
  useLayoutEffect(() => {
    let frame,
      previous = null;
    let clock = {
      elapsed: 0,
      id: current.current.illustration.flightId || 0,
      start: 0,
      instant: true,
      age: 10,
    };
    const publish = (now) => {
      const dt = previous === null ? 0 : (now - previous) / 1000;
      previous = now;
      if (!document.hidden) {
        const { illustration } = current.current;
        clock = advanceGestureClock(
          clock,
          dt,
          illustration.flightId || 0,
          motionRef.current.paused,
          motionRef.current.reducedMotion,
        );
        flightRef.current =
          illustration.result?.id === "physics"
            ? flightPresentation(illustration.result.parameter, clock.age)
            : null;
        if (flightRef.current?.landed && current.current.launcherChallenge.shots.some((shot) => shot.flightId === illustration.flightId && shot.status === "flying"))
          dispatchRef.current({ type: "challenge-landed", flightId: illustration.flightId });
      }
      frame = requestAnimationFrame(publish);
    };
    frame = requestAnimationFrame(publish);
    return () => cancelAnimationFrame(frame);
  }, []);
  const dispatch = (event) => {
    const next = meetingEvent(current.current, studio, event);
    const newOffer = next.noteOffered && !value.noteOffered;
    current.current = next;
    onSessionChange(next);
    if (event.type === "trial-open") setFullConversation(true);
    if (event.type === "interact" && event.id === "greet") setFullConversation(false);
    setRoomFocused((focused) =>
      meetingRoomFocus(focused, event, value.material, next.material, newOffer),
    );
    const pickingUp =
      (event.type === "interact" && event.id === "paper" && !value.noteHeld) ||
      (event.type === "note" && event.action === "take");
    if (
      !pickingUp &&
      (event.type === "interact" || next.material !== value.material)
    )
      room.current?.release();
    if (pickingUp || newOffer || (event.type === "note" && ["return", "offer", "decline"].includes(event.action)))
      requestAnimationFrame(() => room.current?.focus());
    if (
      event.type === "result-card" &&
      ["collect", "carry", "store"].includes(event.action)
    )
      requestAnimationFrame(() => room.current?.focus());
    if (event.type === "result-card" && event.action === "pin")
      requestAnimationFrame(() =>
        dialogRef.current?.querySelector(".meeting-material-board")?.focus(),
      );
    if (next.material === "result" && value.material !== "result")
      requestAnimationFrame(() =>
        dialogRef.current?.querySelector(".meeting-result-reader")?.focus(),
      );
    if (next.material === "paper" && value.material !== "paper")
      requestAnimationFrame(() =>
        dialogRef.current?.querySelector(".meeting-note")?.focus(),
      );
    if (next.material === "experiment" && value.material !== "experiment")
      requestAnimationFrame(() =>
        dialogRef.current
          ?.querySelector('.meeting-material-experiment input[type="range"]')
          ?.focus(),
      );
    if (event.type === "close-material")
      requestAnimationFrame(() => room.current?.focus());
    if (event.type === "trial-open")
      requestAnimationFrame(() => {
        const input = dialogRef.current?.querySelector(`#trial-input-${next.messages.length - 1}`);
        input?.focus();
        const card = input?.closest(".conversation-trial"), log = transcript.current;
        if (card && log) log.scrollTop += card.getBoundingClientRect().top - log.getBoundingClientRect().top - 12;
      });
    if (event.type === "launcher" || event.type === "landing-challenge")
      requestAnimationFrame(() => room.current?.focus());
    if (
      event.type === "quote" ||
      (event.type === "interact" && event.id === "greet")
    )
      requestAnimationFrame(() =>
        document.getElementById("research-message")?.focus(),
      );
  };
  dispatchRef.current = dispatch;
  useEffect(() => {
    const log = transcript.current;
    if (!log) return;
    if (roomDialogue && log.lastElementChild) {
      // Start at the new response, not its final line. Long replies remain
      // independently scrollable without moving the character out of view.
      log.scrollTop += log.lastElementChild.getBoundingClientRect().top - log.getBoundingClientRect().top - 8;
    } else log.scrollTop = log.scrollHeight;
  }, [value.messages.length, roomDialogue]);
  const close = () => {
    if (current.current.launcherActive)
      dispatch({ type: "launcher", action: "close" });
    room.current?.release();
    onClose();
  };
  const returnToRoom = () => {
    setFullConversation(false);
    setRoomFocused(true);
    requestAnimationFrame(() => room.current?.focus());
  };
  const showConversation = () => {
    room.current?.release();
    setFullConversation(false);
    setRoomFocused(false);
    requestAnimationFrame(() => document.getElementById("research-message")?.focus());
  };
  const names = {
    paper: "Shared working note",
    result: "Collected result",
    board: "Shared board",
    experiment:
      studio.id === "chemistry"
        ? "Reaction bench"
        : studio.id === "biology"
          ? "Growth garden"
          : studio.id === "physics"
            ? "Launch bench"
            : "Experiment draft",
  };
  const contextName =
    value.quoteKind === "observation" ? "observation" : "passage";
  const note = workingNote(studio);
  const materialShortcut = conversationMaterialShortcut(value, roomConversationActive);
  const actions = (
    <div className="meeting-inventory" aria-label="Room interactions">
      {ROOM_ACTIONS.map((action) => (
        <button
          key={action.id}
          onClick={() =>
            action.id === "board" && carriedResult
              ? room.current?.resultAction
                ? room.current.resultAction("pin")
                : dispatch({ type: "result-card", action: "pin" })
              : dispatch({ type: "interact", id: action.id })
          }
          aria-label={
            action.id === "paper" && value.noteHeld ? "Read held note"
              : action.id === "paper" && value.noteOffered ? "Take offered note"
                : action.id === "board" && carriedResult ? "Pin result to board" : action.label
          }
          aria-pressed={action.id === "paper" ? value.noteHeld : value.material === action.id}
        >
          <Icon name={action.icon} size={19} />
          <span>{action.id === "paper" && value.noteHeld ? "Note in hand"
            : action.id === "paper" && value.noteOffered ? "Take the note"
              : action.id === "board" && carriedResult ? "Pin this result" : action.hint}</span>
          {value.visited.includes(action.id) && <i aria-label="Explored">·</i>}
        </button>
      ))}
    </div>
  );
  return (
    <StudioDialog
      dialogRef={dialogRef}
      onClose={close}
      className={`rw-session interactive-meeting ${roomFocused ? "is-room-focused" : ""} ${studioView === "first-person" ? "is-first-person" : ""} ${roomDialogue ? "is-room-dialogue" : ""} ${value.material === "experiment" ? "is-equipment-open" : ""}`}
      aria-labelledby="session-title"
      onDismissRequest={() => {
        if (current.current.launcherActive) {
          dispatch({ type: "launcher", action: "close" });
          return;
        }
        if (roomConversationActive) {
          returnToRoom();
          return;
        }
        if (room.current?.consumeEscape()) return;
        if (value.material) dispatch({ type: "close-material" });
        else close();
      }}
    >
      <header className="rw-session-header">
        <button className="rw-text-link" onClick={close}>
          <Icon name="back" size={18} />
          Leave studio
        </button>
        <span>
          <i style={{ background: studio.color }} />
          {studio.label} studio <span>/</span> Interactive preview
        </span>
        <div className="meeting-header-actions">
          <button
            className="meeting-focus-button"
            onClick={roomFocused ? showConversation : returnToRoom}
          >
            <Icon name={roomFocused ? "send" : "compass"} size={16} />
            {roomFocused ? "Show conversation" : roomConversationActive ? "Back to room" : "Focus on the room"}
          </button>
          <button
            className="rw-icon-button"
            aria-label={
              localPaused ? "Resume room motion" : "Pause room motion"
            }
            onClick={() => setLocalPaused(!localPaused)}
          >
            <Icon name={localPaused ? "play" : "pause"} size={17} />
          </button>
          <button
            className="rw-icon-button"
            onClick={close}
            aria-label="Close research session"
          >
            <Icon name="close" />
          </button>
        </div>
      </header>
      <div className="rw-session-layout">
        <section
          className={`meeting-place ${value.material ? "with-material" : ""} ${value.material === "experiment" ? "is-launch-mode" : ""} ${KINETIC_STUDIOS.includes(studio.id) && value.material === "experiment" ? "is-kinetic-mode" : ""}`}
          aria-label="3D research studio"
        >
          <header className="meeting-room-title">
            <div>
              <p className="rw-eyebrow">YOUR LITTLE CORNER OF THE WORLD</p>
              <h2 id="session-title">
                Coffee with {studio.name.split(" ")[0]}.
              </h2>
            </div>
            <span>Explore. Pick things up. Think together.</span>
          </header>
          {robotAsset.ready ? <InteractiveStudio
            studio={studio}
            value={value}
            paused={paused || localPaused}
            reducedMotion={reducedMotion}
            dispatch={dispatch}
            apiRef={room}
            flightRef={flightRef}
            calculation={calculation}
            robotTemplate={robotAsset.template}
            onViewChange={setStudioView}
            actions={actions}
            dialogueActive={roomConversationActive}
          /> : <><div className="meeting-scene-wrap robot-loading" role="status">Preparing your research companion…</div>{actions}</>}
          {robotAsset.failed && <p className="robot-asset-fallback" role="status">Detailed character unavailable. The basic model and all room controls still work.</p>}
          {value.material && (
            <section
              className={`meeting-material meeting-material-${value.material}`}
              aria-label={names[value.material]}
              tabIndex={-1}
            >
              <header className="meeting-material-header">
                <span>
                  <Icon
                    name={
                      value.material === "board"
                        ? "mathematics"
                        : value.material === "paper"
                          ? "paper"
                          : "code"
                    }
                    size={17}
                  />
                  {names[value.material]}
                  <small>Kept during this visit</small>
                </span>
                <button
                  className="rw-icon-button"
                  aria-label="Put material away"
                  onClick={() => dispatch({ type: "close-material" })}
                >
                  <Icon name="close" size={18} />
                </button>
              </header>
              <div className="meeting-material-body">
                {value.material === "board" ? (
                  <SharedBoard
                    board={value.board}
                    dispatch={dispatch}
                    resultCards={value.resultCards}
                  />
                ) : value.material === "result" && selectedResult ? (
                  <div
                    className="meeting-result-reader"
                    tabIndex={-1}
                    aria-label="Collected result reader"
                  >
                    <ResultCard
                      card={selectedResult}
                      pinned={value.board.pinnedResultIds?.includes(
                        selectedResult.id,
                      )}
                      onDiscuss={(card) =>
                        dispatch({
                          type: "quote",
                          kind: "observation",
                          text: resultCardView(card).quote,
                        })
                      }
                    />
                    <button
                      className="take-result-card"
                      onClick={() =>
                        dispatch({
                          type: "result-card",
                          action: "store",
                          id: selectedResult.id,
                        })
                      }
                    >
                      Keep in collection · Q
                    </button>
                  </div>
                ) : value.material === "paper" ? (
                  <article
                    className="meeting-note"
                    tabIndex={-1}
                    aria-label="Working note reader"
                  >
                    <span className="rw-eyebrow">
                      {note.eyebrow}
                    </span>
                    <h3>{note.title}</h3>
                    <p className="meeting-note-intro">
                      Point to a passage. Make it part of the conversation.
                    </p>
                    {note.passages.map((passage, index) => (
                      <section key={passage.title}>
                        <h4>
                          0{index + 1} / {passage.title}
                        </h4>
                        <p>{passage.text}</p>
                        <button
                          onClick={() =>
                            dispatch({ type: "quote", text: passage.text })
                          }
                        >
                          <Icon name="send" size={14} />
                          Ask about this passage
                        </button>
                      </section>
                    ))}
                    <p className="meeting-disclosure">
                      {note.disclosure}
                    </p>
                  </article>
                ) : (
                  <div className="meeting-experiment is-playground">
                    <details className="experiment-writing">
                      <summary>Keep a research idea</summary>
                      <span className="rw-eyebrow">
                        A SMALL TEST, NOT A BIG LEAP
                      </span>
                      <h3>What if we changed one thing?</h3>
                      <label>
                        Your idea
                        <textarea
                          aria-label="Your idea"
                          rows={3}
                          maxLength={4000}
                          value={value.experiment}
                          onChange={(event) =>
                            dispatch({
                              type: "patch",
                              value: {
                                experiment: event.target.value,
                                saved: false,
                              },
                            })
                          }
                          placeholder="What would we compare, and what would change our mind?"
                        />
                      </label>
                      <button
                        className="rw-primary"
                        disabled={!value.experiment.trim()}
                        onClick={() =>
                          dispatch({ type: "patch", value: { saved: true } })
                        }
                      >
                        Save experiment draft <Icon name="arrow" size={16} />
                      </button>
                      <p className="meeting-disclosure" role="status">
                        {value.saved
                          ? "Draft saved for this visit. No experiment has been started."
                          : "Your proposal is a draft. No research worker is connected."}
                      </p>
                    </details>
                    <LocalIllustration
                      studioId={studio.id}
                      paused={paused || localPaused}
                      reducedMotion={reducedMotion}
                      flightRef={flightRef}
                      onSampleChange={(index) =>
                        dispatch({ type: "scrub", index })
                      }
                      onDiscuss={(text) =>
                        dispatch({ type: "quote", text, kind: "observation" })
                      }
                      onCollect={(illustration) =>
                        dispatch({
                          type: "result-card",
                          action: "collect",
                          illustration,
                        })
                      }
                      savedState={value.illustration}
                      onParameterChange={(parameter) =>
                        dispatch({
                          type: "patch",
                          value: {
                            illustration: {
                              ...current.current.illustration,
                              parameter,
                            },
                          },
                        })
                      }
                      calculation={calculation}
                    />
                  </div>
                )}
              </div>
            </section>
          )}
        </section>
        <aside className="rw-conversation" aria-label={roomDialogue ? "In-room dialogue" : "Conversation workspace"}>
          <div className="rw-conversation-title">
            <span
              className="rw-conversation-avatar"
              style={{ "--studio-color": studio.color }}
            >
              <Icon name={studio.id} size={22} />
            </span>
            <div>
              <h3>{studio.name}</h3>
              <p>
                {roomDialogue ? "Scripted preview" : <>{value.material
                  ? `Together at the ${value.material === "paper" ? "working note" : value.material}`
                  : "Here with you"}{" "}
                · Scripted preview</>}
              </p>
            </div>
            {roomDialogue ? <button className="meeting-history-button" aria-label="Full conversation" onClick={() => {
              setFullConversation(true);
              requestAnimationFrame(() => transcript.current?.focus());
            }}>History</button> : <span className="rw-conversation-dot" />}
          </div>
          <div
            ref={transcript}
            className="rw-transcript"
            role="log"
            aria-label="Research conversation"
            aria-live="polite"
            tabIndex={0}
          >
            {conversationEntries(value.messages, roomDialogue).map(({ message, index: i }) => (
              <div key={i} className={`rw-message rw-message-${message.role}`}>
                <span>
                  {message.role === "human"
                    ? "YOU"
                    : message.role === "calculation"
                      ? "BROWSER CALCULATION"
                      : studio.name.toUpperCase()}
                </span>
                {message.quote && <blockquote>{message.quote}</blockquote>}
                <p>{message.text}</p>
                {message.trial && <ConversationTrial trial={message.trial} index={i} calculation={calculation} dispatch={dispatch} />}
              </div>
            ))}
          </div>
          <div className="meeting-conversation-shortcuts">
            <button aria-label={materialShortcut.label} onClick={() => {
              if (materialShortcut.event.type === "result-card" && room.current?.resultAction)
                room.current.resultAction("read");
              else dispatch(materialShortcut.event);
            }}>
              {roomDialogue ? materialShortcut.shortLabel : materialShortcut.label}
            </button>
            <button aria-label="Let’s use the board" onClick={() => dispatch({ type: "interact", id: "board" })}>
              {roomDialogue ? "Board" : "Let’s use the board"}
            </button>
            <button
              aria-label="Try a browser experiment"
              onClick={() => dispatch({ type: "trial-open" })}
            >
              {roomDialogue ? "Experiment" : "Try a browser experiment"}
            </button>
          </div>
          {value.quote && (
            <div className="meeting-quote-context">
              <div>
                <Icon name="paper" size={15} />
                <span>Asking about this {contextName}</span>
                <button
                  aria-label={`Remove ${contextName} context`}
                  onClick={() => dispatch({ type: "quote", text: "" })}
                >
                  ×
                </button>
              </div>
              <blockquote>{value.quote}</blockquote>
              <button
                onClick={() =>
                  dispatch({
                    type: "send",
                    text: `Explain this ${contextName} and its assumptions.`,
                  })
                }
              >
                Discuss this {contextName} <Icon name="arrow" size={14} />
              </button>
            </div>
          )}
          <form
            className="rw-compose"
            onSubmit={(event) => {
              event.preventDefault();
              dispatch({ type: "send" });
            }}
          >
            <label htmlFor="research-message">Your question</label>
            <div>
              <textarea
                id="research-message"
                rows={roomDialogue ? 1 : 2}
                maxLength={4000}
                value={value.draft}
                onChange={(event) =>
                  dispatch({
                    type: "patch",
                    value: { draft: event.target.value },
                  })
                }
                placeholder={
                  value.quote
                    ? `What about this ${contextName}?`
                    : "Follow a thought…"
                }
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    dispatch({ type: "send" });
                  }
                }}
              />
              <button
                type="submit"
                disabled={!value.draft.trim()}
                aria-label="Send message"
              >
                <Icon name="send" size={19} />
              </button>
            </div>
            <p>{roomDialogue ? "Scripted replies · No model connected" : "Scripted replies · Visit-only memory · No model connected"}</p>
          </form>
        </aside>
      </div>
    </StudioDialog>
  );
}

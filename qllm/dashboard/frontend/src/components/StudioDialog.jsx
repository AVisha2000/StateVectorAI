import { useEffect } from "react";

// Keyboard step-back belongs to the studio. Native close remains authoritative:
// a browser can issue a non-cancellable close request after repeated Escape.
export default function StudioDialog({ dialogRef, onDismissRequest, onClose, children, ...props }) {
  useEffect(() => {
    const dialog = dialogRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.showModal();
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
    };
  }, [dialogRef]);

  return <dialog {...props} ref={dialogRef}
    onKeyDown={(event) => {
      if (event.key !== "Escape" || event.defaultPrevented || event.nativeEvent.isComposing) return;
      event.preventDefault();
      event.stopPropagation();
      if (!event.repeat) onDismissRequest();
    }}
    onCancel={(event) => {
      if (event.target !== event.currentTarget || !event.cancelable) return;
      event.preventDefault();
      onDismissRequest();
    }}
    onClose={(event) => {
      // Ignore descendant dialogs and a queued cleanup close after StrictMode
      // has reopened this same element. Final unmount needs no parent update.
      if (event.target === event.currentTarget && event.currentTarget.isConnected && !event.currentTarget.open)
        onClose();
    }}
  >{children}</dialog>;
}

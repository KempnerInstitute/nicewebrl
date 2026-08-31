async function isFullscreen() {
  return document.fullscreenElement != null;
} 

async function getImageSeenTime() {
  return window.imageSeenTime;
} 

// Function to send ping to server every 30 seconds
async function pingServer() {
  console.log('Starting ping loop');
  while (true) {
    try {
      const message = `Ping ${Math.floor(Math.random() * 10000)}`;
      // Await the asynchronous emitEvent call
      await emitEvent('ping', { message: message });
      console.log(`Ping sent: ${message}`);
    } catch (err) {
      console.error('Error pinging server:', err);
    }
    // Wait for 30 seconds before sending the next ping
    seconds = 30
    try {
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    } catch (err) {
      console.error('Error in ping loop:', err);
    }
  }
}

// Function to send ping to server every 30 seconds
async function userMessage(
  called_by_room_id,
  called_by_user_id,
  event,
  stage,
  message) {
  console.log(event, stage, message)
  await emitEvent(event, {
    called_by_room_id: called_by_room_id,
    called_by_user_id: called_by_user_id,
    event: event,
    stage: stage,
    message: message,
  });
}


// Function to select next image based on environment actions
async function updateEnvironment(room_id, action_key) {
  console.log(room_id, action_key)
  if (room_id !== window.room) {
    console.log('room_id !== window.room', room_id, window.room)
    return;
  }
  if (action_key == 'starting' || action_key == '') {
    return;
  }
  if (!window.require_fullscreen || await isFullscreen() ) {
    next_state = window.next_states[action_key];
    var imgElement = document.getElementById('stateImage');
    if (imgElement !== null) {
      imgElement.src = next_state;
    }
    window.next_imageSeenTime = new Date();
    console.log('set new image');
    await emitEvent('update_environment', {
      called_by_room_id: window.room,
      action_key: action_key,
    });
  }
}

document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // Start pinging the server once the DOM content is fully loaded
  ////////////////
  pingServer();

  ////////////////
  // remove default behavior
  ////////////////
  window.debug = 0;
  window.require_fullscreen = false;
  window.accept_keys = false;
  window.next_states = null;
  window.stage = null;
  window.room = null;
  window.user_id = null;

  ////////////////
  // Prevent spacebar from toggling fullscreen
  ////////////////
  document.addEventListener('keydown', function (event) {
    // Check if the key pressed is spacebar
    if (event.key === " " || event.code === "Space") {
      // Prevent the default action (toggling fullscreen)
      event.preventDefault();
    }
  }, true); // Using capturing phase to catch the event before other handlers

  ////////////////
  // how to handle key presses?
  ////////////////
  document.addEventListener('keydown', async function (event) {
    // Prevent default behavior for arrow keys
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
    }

    // Handle key presses
    console.log(event.key);
    //console.log('window.next_states !== null', window.next_states !== null)
    //console.log('window.accept_keys', window.accept_keys)
    //console.log('event.key in window.next_states', event.key in window.next_states)
    //if (window.next_states !== null && window.accept_keys && event.key in window.next_states) {
    //  if (!window.require_fullscreen || await isFullscreen() ) {
    //    next_state = window.next_states[event.key];
    //    var imgElement = document.getElementById('stateImage');
    //    if (imgElement !== null) {
    //      imgElement.src = next_state;
    //    }
    //    window.next_imageSeenTime = new Date();
    //    console.log('set new image');
    //  }
    //  // Record the current time when the keydown event occurs
      //var keydownTime = new Date();
      //// Await the asynchronous emitEvent call
      //await emitEvent('key_pressed', {
      //  key: event.key,
      //  keydownTime: keydownTime,
      //  imageSeenTime: window.imageSeenTime,
      //  client: window.user_id,
      //  room: window.room,
      //});
    //}
    //// REGULAR KEY PRRESS
    //else {
    var keydownTime = new Date();
    await emitEvent('key_pressed', {
      key: event.key,
      client: window.user_id,
      room: window.room,
      imageSeenTime: window.imageSeenTime,
      keydownTime: keydownTime,
    });
    //}

  });
})
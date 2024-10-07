window.addEventListener('DOMContentLoaded', () => {
    const videoStream = document.getElementById('video-stream');

    // Function to fetch the video stream
    function fetchStream() {
        videoStream.src = "/video_feed";
    }

    fetchStream();
});

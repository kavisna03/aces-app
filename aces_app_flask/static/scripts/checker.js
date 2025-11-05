// JavaScript code for the checker form in the Checker section
// Handles form validation and resetting of the appropriateness result when the users attempts to submit the form

document.getElementById('checker-form').addEventListener('submit', function(event) {
  const videoFile = document.getElementsByName('video-file')[0];

  // Display alert message if video file is not uploaded
  if (!videoFile.files.length)
  {
    // Don't allow form to be submitted
    event.preventDefault();
    alert('Please upload a video file.');  
  } 
  // Display alert message if the uploaded file's format does not match with any valid video file formats
  else
  {
    const videoFileType = videoFile.files[0].type;
  
    if(!videoFileType.startsWith('video/'))
    {
      // Don't allow form to be submitted
      event.preventDefault();
      alert('Please upload a valid video file.');  
    }
  }

  const criterion1 = document.querySelector('input[name="language-complexity"]:checked');
  const criterion2 = document.querySelector('input[name="presentation-complexity"]:checked');

  // Display alert message if none of the criteria are selected
  if (!criterion1 && !criterion2)
  {
    // Don't allow form to be submitted
    event.preventDefault();
    alert('Please select one or more criteria.');  
  }

  // Reset appropriateness result bar (if result is shown) when the user submits the form to evaluate the next video
  const apprResultBar = document.getElementsByClassName('appr-result-bar')[0];

  if (apprResultBar)
  {
    apprResultBar.innerHTML = "";
  }
});



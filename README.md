# farmer-life

This data was collected from text ads found on twelve websites that deal with various farm animal related topics.  Information from the ad creative and the ad landing page is included.  The binary labels are based on whether or not the content owner approves of the ad.

For each ad, we include the words on the ad creative and the words from the landing page.  Each word from the creative is given a prefix
of 'ad-'.  Title and header HTML markups are noted in a similar way in the text of the landing page.  We have already performed stemming and
stop word removal.  Each ad is on a single line.  The first word in the line is the label of the instance.  It is 1 for accepted ads and -1 for rejected ads.

We have also included a straightforward bag-of-words representation of our data.  We use the SVMlight sparse vector format.  The first value
is the label followed by every nonzero attribute.  Each of these attributes is encoded as index:value.  This is the representation used for the relevant paper cited below.

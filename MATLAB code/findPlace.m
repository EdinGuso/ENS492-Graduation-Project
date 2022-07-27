function col = findPlace(test_g,bookKeeping)
    r = 1;
    term = false;
    col = 0;
    sz = size(bookKeeping);
    while r < sz(2) && ~term
        if(round(bookKeeping(1,r)-test_g,3) == 0)
            col = r;
            term = true;
        end
        if ~term
        r = r+1;
        end
    end
    
end
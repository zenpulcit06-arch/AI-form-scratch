program main
    use, intrinsic :: iso_fortran_env
    use Layer
    use Logistic_regression, only: sigmoid
    implicit none
    integer(int64) :: n_feature,sample_size, neurons1, neurons2, iteration,i,j,mode
    real(real64) :: learning_rate,loss
    real(real64), allocatable :: x(:,:), y(:,:) 
    character(len = 1000) :: filename
    type(network) :: net

    print *, "Enter no. of feature"
    read(*,*) n_feature

    print *, "Enter sample size"
    read(*,*) sample_size

    print *, "Enter no. of neurons in hidden layer1"
    read(*,*) neurons1

    print *, "Enter no. of neurons in hidden layer2"
    read(*,*) neurons2

    print *, 'Enter no. of iteration'
    read(*,*) iteration

    print *, 'Enter learning rate'
    read(*,*) learning_rate

    if (allocated(x)) deallocate(x)
    allocate(x(sample_size,n_feature))

    if (allocated(y)) deallocate(y)
    allocate(y(sample_size, neurons2))

    mode = 0
    print *, 'Enter 1 for manual entry or 2 for CSV entry:'
    read(*,*) mode

    if (mode == 2) then
        print *, 'Enter Filename:'
        read(*,*) filename
        print *, 'Reading data from ', trim(filename), '...'
        open(unit=10, file=trim(filename), status='old', action='read')
        do i = 1, sample_size
            read(10, *) x(i, :), y(i, 1)
        end do
        close(10)
        print *, 'Data loaded successfully.'
    else
        do i = 1, sample_size
            print *, '--- Sample ', i, ' ---'
            do j = 1, n_feature
                print *, 'Enter feature ', j, ':'
                read(*,*) x(i, j)
            end do
            print *, 'Enter target label y (0 or 1):'
            read(*,*) y(i,1)
        end do
    end if

    call intialize(net,neurons1,neurons2,n_feature,sample_size)
    call fit(net,x,y,neurons1,neurons2,sample_size,learning_rate &
            , iteration,loss, n_feature)
    
    print *, 'Loss =',loss

end program main